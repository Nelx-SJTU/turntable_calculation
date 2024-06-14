import cv2
import numpy as np
from scipy.optimize import least_squares


class FixedSizeArray:
    def __init__(self, max_len=100, data_size=(0, 3), axis=0):
        self.max_len = max_len
        self.data_size = data_size
        self.axis = axis
        self.data = np.empty(self.data_size)

    def append(self, value):
        if len(self.data) < self.max_len:
            self.data = np.append(self.data, value, axis=self.axis)
        else:
            self.data = np.append(self.data[1:], value, axis=self.axis)

    def get_array(self):
        return self.data


class FixedSizeDict:
    def __init__(self, max_len=100, data_size=(0, 3)):
        self.max_len = max_len
        self.data_size = data_size
        self.data = {}

    def append(self, key, value):
        if key not in self.data:
            # Initialize as empty array
            self.data[key] = np.empty(self.data_size)
        if self.data[key].shape[0] < self.max_len:
            self.data[key] = np.append(self.data[key], [value], axis=0)
        else:
            self.data[key] = np.append(self.data[key][1:], [value], axis=0)

    def get_dict(self):
        return self.data

    def reset(self):
        self.data = {}


class ArucoTag:
    def __init__(self, tag_name, track_len, aruco_dict_type):
        self.tag_name = tag_name
        self.track_len = track_len  # The length of tracking the pose of aruco tag
        self.aruco_dict_type = aruco_dict_type

        self.data = FixedSizeDict(self.track_len)

    def update(self, rvec, tvec):
        # rotation vector in camera coordinate
        self.data.append('rv_c', rvec)  # If cannot detect certain tag, set it as None
        # translation vector in camera coordinate
        self.data.append('tv_c', tvec)  # If cannot detect certain tag, set it as None

    def get_data(self):
        return self.data

    def print_info(self):
        print('-------------ArucoTag INFO-------------')
        print('ARUCO TAG NAME : ', self.tag_name)
        print('TRACK LENGTH : ', self.track_len)
        print('data : ', self.data.get_dict())


class ArucoTagSet:
    def __init__(self, set_name, track_len, aruco_tags_id_list, aruco_dict_type):
        self.set_name = set_name
        self.track_len = track_len
        self.aruco_tags_id_list = aruco_tags_id_list
        self.aruco_dict_type = aruco_dict_type

        # ArucoTagSet parameters

        # Initialize aruco tags
        self.data = {name: ArucoTag(name, self.track_len, self.aruco_dict_type) for name in self.aruco_tags_id_list}

    def update(self, poses):
        for key, (rvec, tvec) in poses.items():
            self.data[key].update(rvec[0], tvec[0])

    def get_data(self):
        return self.data

    def print_info(self):
        print('------------------------ArucoTagSet INFO------------------------')
        print('ARUCO SET NAME : ', self.set_name)
        print('TRACK LENGTH : ', self.track_len)
        print('ARUCO TAG IDs : ', self.aruco_tags_id_list)
        for key in self.aruco_tags_id_list:
            print('Start getting information about Aruco Tag :', key)
            self.data[key].print_info()


class ArucoDetector:
    def __init__(self, aruco_dict_type, aruco_tags_id_list, marker_size, camera_matrix, dist_coeffs, draw=False):
        # Load the dictionary that was used to generate the markers.
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        self.aruco_tags_id_list = aruco_tags_id_list
        self.marker_size = marker_size
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.draw = draw
        if self.draw:
            cv2.namedWindow('Detected Aruco markers', 0)

    def detect_aruco_poses(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect the markers in the image.
        corners, ids, _ = cv2.aruco.detectMarkers(image_gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            # Save the results in a dict
            poses = {id[0]: (rvec, tvec) for id, rvec, tvec in zip(ids, rvecs, tvecs)}

            # Sort the dict <pose> in correct order as self.aruco_tags_id_list
            # If id in self.aruco_tags_id_list does not exist, return None
            poses = {id: poses[id] if id in poses else ([[None, None, None]], [[None, None, None]])
                     for id in self.aruco_tags_id_list}

            # Check whether all the id in self.aruco_tags_id_list are founded
            found_all = all([id in poses for id in self.aruco_tags_id_list])

            if self.draw:
                image_with_aruco = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
                cv2.imshow('Detected Aruco markers', image_with_aruco)
                cv2.waitKey(1)
            return poses, found_all
        else:
            return None, False


class TurnTable:
    def __init__(self, aruco_tags_id_list, turntable_diameter, camera_coordinate_c, trajectory_track_len):
        self.origine_aruco_tags_id_list = aruco_tags_id_list
        self.aruco_tags_id_list = aruco_tags_id_list
        self.turntable_diameter = turntable_diameter
        self.camera_coordinate_c = camera_coordinate_c
        self.camera_coordinate_t = None
        self.trajectory_track_len = trajectory_track_len

        # Calibration points
        self.calibration_data_list_rotation_filtered = []
        self.calibration_data_list_translation_filtered = []

        # Rotation matrix (from camera-centered coordinate to turntable-centered coordinate)
        self.camera_to_turntable_R = None

        # Fitting results using calibration points
        self.fitted_center = None
        self.fitted_radius = None
        self.rotation_axis = None

        # Trajectory of aruco tags
        self.trajectory = {id: FixedSizeDict(self.trajectory_track_len) for id in self.aruco_tags_id_list}

        # Zero reference
        self.zero_ref_pos = FixedSizeDict(max_len=1, data_size=(0, 3))
        self.zero_ref_R = FixedSizeDict(max_len=1, data_size=(0, 3, 3))
        self.zero_ref_rvec = FixedSizeDict(max_len=1, data_size=(0, 3))

        # Current Position
        self.current_positions_on_circle = FixedSizeDict(max_len=1, data_size=(0, 3))
        self.current_angles = FixedSizeDict(max_len=1, data_size=(0, 1))

        # Angle filter
        # self.basic_angle_filter = BasicAngleFilter(filter_delta_angle=np.deg2rad(10))

    def set_zero_reference_with_single_frame(self):
        all_points_detected = True
        for id in self.aruco_tags_id_list:
            if np.all(self.trajectory[id].get_dict()['pos_t'][-1] == [None, None, None]):
                all_points_detected = False
                print('Cannot set 0 reference as not all the tags are detected.')
                break
            else:
                # Position of 0 ref
                self.zero_ref_pos.append(id, self.trajectory[id].get_dict()['pos_t'][-1])

                # Rotation materix of 0 ref
                _zero_ref_R, _ = cv2.Rodrigues(self.trajectory[id].get_dict()['rv_c'][-1].astype(np.float64))
                self.zero_ref_R.append(id, _zero_ref_R)

                # rvec of 0 ref
                self.zero_ref_rvec.append(id, self.trajectory[id].get_dict()['rv_c'][-1])
        if all_points_detected:
            print('0 Reference set to', self.zero_ref_pos.get_dict())
        else:
            self.zero_ref_pos.reset()
        return all_points_detected

    def set_zero_reference_with_multiple_frames(self):
        all_points_detected = True
        self.aruco_tags_id_list = []
        for id_cnt, id in enumerate(self.origine_aruco_tags_id_list):
            zero_ref_pos_list = []
            zero_ref_R_list = []
            zero_ref_rvec = []
            for cnt in range(self.trajectory_track_len):
                if np.all(self.trajectory[id].get_dict()['pos_t'][cnt] != [None, None, None]):
                    # Position of 0 ref
                    # zero_ref_pos_list.append(self.trajectory[id].get_dict()['pos_t'][cnt])
                    zero_ref_pos_list.append(find_closest_point_on_circle(
                        A=self.trajectory[id].get_dict()['pos_t'][cnt],
                        O=[0, 0, 0],
                        R=self.fitted_radius[id_cnt],
                        n=[0, 0, 1]))

                    # Rotation materix of 0 ref
                    _zero_ref_R, _ = cv2.Rodrigues(self.trajectory[id].get_dict()['rv_c'][-1].astype(np.float64))
                    zero_ref_R_list.append(_zero_ref_R)

                    # rvec of 0 ref
                    zero_ref_rvec.append(self.trajectory[id].get_dict()['rv_c'][cnt])
            if len(zero_ref_pos_list) < 5:
                print('Cannot detect TAG_id:', id, ' in the previous 5 frames.')
            else:
                self.aruco_tags_id_list.append(id)
                self.zero_ref_pos.append(id, np.mean(zero_ref_pos_list, axis=0))
                self.zero_ref_R.append(id, np.mean(np.array(zero_ref_R_list), axis=0))
                self.zero_ref_rvec.append(id, np.mean(zero_ref_rvec, axis=0))

        if len(self.origine_aruco_tags_id_list) - len(self.aruco_tags_id_list) > 2:
            all_points_detected = False

        # Reset the self.aruco_tags_id_list
        self.aruco_tags_id_list = np.array(self.aruco_tags_id_list)
        return all_points_detected

    def update(self, id, rotation_data, translation_data):
        self.trajectory[id].append('pos_c', translation_data)
        if np.all(rotation_data == [None, None, None]):
            self.trajectory[id].append('pos_t', [None, None, None])
            self.trajectory[id].append('rv_c', [None, None, None])
        else:
            self.trajectory[id].append('pos_t', np.dot(self.camera_to_turntable_R.T, (translation_data - self.fitted_center)))
            self.trajectory[id].append('rv_c', rotation_data)

    def circular_rotation_fitting(self, aruco_tag_set):
        fitted_center_list = np.empty((0, 3))
        fitted_radius_list = np.empty((0, 0))
        rotation_axis_list = np.empty((0, 3))

        for tag_cnt, tag in enumerate(self.aruco_tags_id_list):
            rotation_data = aruco_tag_set.get_data()[tag].get_data().get_dict()['rv_c']
            translation_data = aruco_tag_set.get_data()[tag].get_data().get_dict()['tv_c']

            # Create an array of boolean indices marking each row as equal to [None, None, None]
            rotation_data_bool_index = ~np.all(rotation_data == [None, None, None], axis=1)
            translation_data_bool_index = ~np.all(translation_data == [None, None, None], axis=1)

            # Filter out rows equal to [None, None, None] using an array of Boolean indexes
            calibration_data_rotation_filtered = rotation_data[rotation_data_bool_index]
            calibration_data_translation_filtered = translation_data[translation_data_bool_index]
            calibration_data_rotation_filtered = [np.asarray(rvec, dtype=np.float32) for rvec in
                                                  calibration_data_rotation_filtered]
            calibration_data_translation_filtered = [np.asarray(tvec, dtype=np.float32) for tvec in
                                                     calibration_data_translation_filtered]
            self.calibration_data_list_rotation_filtered.append(calibration_data_rotation_filtered)
            self.calibration_data_list_translation_filtered.append(calibration_data_translation_filtered)

            rotation_matrices = np.array([cv2.Rodrigues(rvec)[0] for rvec in self.calibration_data_list_rotation_filtered[tag_cnt]])  # Shape=(n,3,3)

            transformation_matrices = [create_transformation_matrix(rotation_matrices[i], self.calibration_data_list_translation_filtered[tag_cnt][i])
                                       for i in range(len(rotation_matrices))]

            # Get the position of the calibration plate in the world coordinate system.
            tag_positions_c = [self.calibration_data_list_translation_filtered[tag_cnt][i]
                               for i in range(len(self.calibration_data_list_translation_filtered[tag_cnt]))]

            # Least Squares Fitting of Circles
            fitted_center_tmp, fitted_radius_tmp, rotation_axis_tmp = fit_circle_3d(tag_positions_c)
            fitted_center_list = np.append(fitted_center_list, np.array([fitted_center_tmp]), axis=0)
            fitted_radius_list = np.append(fitted_radius_list, fitted_radius_tmp)
            rotation_axis_list = np.append(rotation_axis_list, np.array([rotation_axis_tmp]), axis=0)

        self.fitted_center = np.mean(fitted_center_list, axis=0)
        self.fitted_radius = fitted_radius_list
        self.rotation_axis = np.mean(rotation_axis_list, axis=0)

        print('Circular rotation fitting complete.')
        print(f"Fitted center: {self.fitted_center}")
        print(f"Fitted radius: {self.fitted_radius}")
        print(f"Rotation axis: {self.rotation_axis}")
        print("Calibration succeed. ")

        # return self.fitted_center, self.fitted_radius, self.rotation_axis

    def camera_coordinate_transform(self):
        # Calculate coordinate transform parameters (Change to turntable centered coordinate)
        camera_translation_vector = self.camera_coordinate_c - self.fitted_center

        # Calculate new base of x-axis (in turntable coordinate)
        new_x_axis_base = camera_translation_vector - np.dot(camera_translation_vector,
                                                             self.rotation_axis) * self.rotation_axis
        new_x_axis_base = new_x_axis_base / np.linalg.norm(new_x_axis_base)

        # Calculate new base of y-axis (in turntable coordinate)
        new_y_axis_base = np.cross(self.rotation_axis, new_x_axis_base)
        new_y_axis_base = new_y_axis_base / np.linalg.norm(new_y_axis_base)

        # Calculate the rotation matrix
        self.camera_to_turntable_R = np.array([new_x_axis_base, new_y_axis_base, self.rotation_axis]).T

        # Calculate new camera coordinate
        self.camera_coordinate_t = np.dot(self.camera_to_turntable_R.T, (self.camera_coordinate_c - self.fitted_center))

    def calculate_rotation_angle(self):
        for id_cnt, id in enumerate(self.aruco_tags_id_list):
            if np.all(self.trajectory[id].get_dict()['pos_t'][-1] != [None, None, None]):
                self.current_positions_on_circle.append(id,
                                                        find_closest_point_on_circle(
                                                            A=self.trajectory[id].get_dict()['pos_t'][-1],
                                                            O=[0, 0, 0],
                                                            R=self.fitted_radius[id_cnt],
                                                            n=[0, 0, 1]))

                # METHOD 1: use position to calculate angle
                current_angle_p = angle_between_vectors(u=self.current_positions_on_circle.get_dict()[id],
                                                        v=self.zero_ref_pos.get_dict()[id][0],
                                                        n=[0, 0, 1])
                current_angle_p = current_angle_p[0]

                # # METHOD 2: use transform matrix to calculate angle
                # current_R, _ = cv2.Rodrigues(self.trajectory[id].get_dict()['rv_c'][-1].astype(np.float64))
                # R_rel = current_R @ self.zero_ref_R.get_dict()[id].T
                #
                # current_angle_r = rotation_matrix_to_angle(R_rel)
                # current_angle_r = current_angle_r[0]
                # if current_angle_p*current_angle_r < 0:
                #     current_angle_r = -current_angle_r

                # METHOD 3: use quaternion to calculate angle
                current_angle_q = calculate_rotation_angle_from_quaternions(self.zero_ref_rvec.get_dict()[id][0],
                                                                            self.trajectory[id].get_dict()['rv_c'][-1])
                if current_angle_p*current_angle_q < 0:
                    current_angle_q = -current_angle_q

                # print('----------------------------------')
                # print(np.degrees(current_angle_p))
                # print(np.degrees(current_angle_q))

                # filtered_angle = self.basic_angle_filter.run_filter(id, [np.mean([current_angle_p, current_angle_q])])
                # self.current_angles.append(id, [np.mean([current_angle_p, current_angle_q])])

                # filtered_angle = self.basic_angle_filter.run_filter(id, [current_angle_p])
                # print('[filtered_angle]', filtered_angle)

                # Filter some strange values
                if np.abs(current_angle_p - current_angle_q) < 0.05:
                    self.current_angles.append(id, [np.mean([current_angle_p, current_angle_q])])
                else:
                    self.current_angles.append(id, [current_angle_p])
            else:
                self.current_positions_on_circle.append(id, np.array([None, None, None]))
                self.current_angles.append(id, np.array([None]))

        # current_angle = np.mean([_data for _id, _data in self.current_angles.get_dict().items()
        #                          if np.all(self.current_angles.get_dict()[_id] != [None])])
        
        # The case where len(self.current_angle) > 1
        current_angle = np.mean([_data[-1] for _id, _data in self.current_angles.get_dict().items()
                                 if np.all(self.current_angles.get_dict()[_id][-1] != None)])
        
        return self.current_positions_on_circle, current_angle

    def plot_rotation(self, ax, currect_position_on_circle):
        for id_cnt, id in enumerate(self.aruco_tags_id_list):

            if np.all(currect_position_on_circle.get_dict()[id][0] != [None, None, None]):
                ax.plot([0, currect_position_on_circle.get_dict()[id][0, 0]],
                        [0, currect_position_on_circle.get_dict()[id][0, 1]],
                        [0, currect_position_on_circle.get_dict()[id][0, 2]],
                        color='green', linestyle='--')

    def plot_points(self, ax, plot_length):
        ax.cla()

        # Plot the camera coordinate to (0,0,0)
        ax.scatter(self.camera_coordinate_t[0], self.camera_coordinate_t[1], self.camera_coordinate_t[2],
                   color='red', marker='o', label='Camera Origin')

        # Plot the fitted rotation center
        ax.scatter(0, 0, 0, color='green', marker='x', label='Rotation Center')

        # Plot the predicted circle
        plot_circle(ax, np.array([0, 0, 0]), self.turntable_diameter, np.array([0, 0, 1]))

        # Plot the rotation axis
        normal_line = np.array([np.array([0, 0, 0]) - np.array([0, 0, 1]) * self.turntable_diameter,
                                np.array([0, 0, 0]) + np.array([0, 0, 1]) * self.turntable_diameter])
        ax.plot(normal_line[:, 0], normal_line[:, 1], normal_line[:, 2], color='green',
                                 label='Rotation Axis (Normal Vector)')

        # Plot the zero degree
        # print('----------------------------------------------')
        # print(self.zero_ref_pos.get_dict())
        if len(self.zero_ref_pos.get_dict()) > 0:
            for id in self.aruco_tags_id_list:
                ax.scatter(self.zero_ref_pos.get_dict()[id][0, 0], self.zero_ref_pos.get_dict()[id][0, 1],
                           self.zero_ref_pos.get_dict()[id][0, 2], color='red', marker='x', label='0 Ref')

        for id in self.aruco_tags_id_list:
            tag_positions_t = self.trajectory[id].get_dict()['pos_t']

            # Create an array of boolean indices marking each row as equal to [None, None, None]
            tag_positions_t_index = ~np.all(tag_positions_t == [None, None, None], axis=1)

            # Filter out rows equal to [None, None, None] using an array of Boolean indexes
            tag_positions_t_filtered = tag_positions_t[tag_positions_t_index]

            if len(tag_positions_t_filtered > 0):
                ax.scatter(tag_positions_t_filtered[-plot_length:, 0], tag_positions_t_filtered[-plot_length:, 1],
                           tag_positions_t_filtered[-plot_length:, 2],
                           color='blue', label='Calibrationboard')

                ax.scatter(tag_positions_t_filtered[-1, 0], tag_positions_t_filtered[-1, 1],
                           tag_positions_t_filtered[-1, 2],
                           color='blue', s=40, label='Calibrationboard')

        # Set the XYZ axis scale the same
        # ax.set_xlim([-60, 60])
        # ax.set_ylim([-60, 60])
        # ax.set_zlim([-60, 60])
        ax.axis('equal')

    def plot_corners(self):
        return 0

    def plot_calibration_result(self, camera_center_ax):
        # Plot the calibration result
        camera_center_ax.cla()

        # Plot the camera coordinate to (0,0,0)
        camera_center_ax.scatter(self.camera_coordinate_c[0], self.camera_coordinate_c[1], self.camera_coordinate_c[2],
                                 color='red', marker='o', label='Camera Origin')

        # Plot the fitted rotation center
        camera_center_ax.scatter(self.fitted_center[0], self.fitted_center[1], self.fitted_center[2],
                                 color='green', marker='x', label='Rotation Center')

        # Plot the predicted circle
        plot_circle(camera_center_ax, self.fitted_center, self.turntable_diameter, self.rotation_axis)

        # Plot the rotation axis
        normal_line = np.array([self.fitted_center - self.rotation_axis * self.turntable_diameter,
                                self.fitted_center + self.rotation_axis * self.turntable_diameter])
        camera_center_ax.plot(normal_line[:, 0], normal_line[:, 1], normal_line[:, 2], color='green',
                              label='Rotation Axis (Normal Vector)')

        # Plot the position of calibration plate
        for tag_cnt, tag in enumerate(self.aruco_tags_id_list):
            tag_positions_c = np.array([self.calibration_data_list_translation_filtered[tag_cnt][i]
                                        for i in range(len(self.calibration_data_list_translation_filtered[tag_cnt]))])

            camera_center_ax.scatter(tag_positions_c[:, 0], tag_positions_c[:, 1],
                                     tag_positions_c[:, 2],
                                     color='blue', label='Calibrationboard')

        # Set the XYZ axis scale the same
        # camera_center_ax.set_xlim([-30, 30])
        # camera_center_ax.set_ylim([-30, 30])
        # camera_center_ax.set_zlim([-60, 60])
        camera_center_ax.axis('equal')


class BasicAngleFilter:
    def __init__(self, filter_delta_angle):
        self.filter_delta_angle = filter_delta_angle
        self.angle = FixedSizeDict(max_len=2, data_size=(0, 1))

    def run_filter(self, id, angle):
        if id in self.angle.get_dict():
            if angle - self.angle.get_dict()[id][-1] > self.filter_delta_angle:
                print('use filter')
                predict_angle = 2 * self.angle.get_dict()[id][-1] - self.angle.get_dict()[id][-2]
                self.angle.append(id, predict_angle)
                return predict_angle
            else:
                self.angle.append(id, angle)
                return angle
        else:
            self.angle.append(id, [0.0])
            self.angle.append(id, [0.0])
            return [0.0]

    def get_data(self):
        return self.angle


class KalmanFilterAruco:
    def __init__(self, dt, q_rvec, q_tvec, r_rvec, r_tvec):
        self.dt = dt
        self.q_rvec = q_rvec
        self.q_tvec = q_tvec
        self.r_rvec = r_rvec
        self.r_tvec = r_tvec

        # 状态转移矩阵
        self.F = np.block([
            [np.eye(3), dt * np.eye(3), np.zeros((3, 6))],
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))],
            [np.zeros((3, 6)), np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 9)), np.eye(3)]
        ])

        # 测量矩阵
        self.H = np.block([
            [np.eye(3), np.zeros((3, 9))],
            [np.zeros((3, 6)), np.eye(3), np.zeros((3, 3))]
        ])

        # 过程噪声协方差
        # self.Q = q * np.eye(12)
        self.Q = np.block([
            [q_rvec * np.eye(3), np.zeros((3, 9))],
            [np.zeros((3, 3)), q_rvec * np.eye(3), np.zeros((3, 6))],
            [np.zeros((3, 6)), q_tvec * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 9)), q_tvec * np.eye(3)]
        ])

        # 测量噪声协方差
        # self.R = r * np.eye(6)
        self.R = np.block([
            [r_rvec * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), r_tvec * np.eye(3)]
        ])

        # 初始化滤波器字典
        self.filters = {}

    def init_filter(self, id):
        """ 初始化滤波器的状态和协方差 """
        x_est = np.zeros(12)
        P_est = np.eye(12)
        self.filters[id] = {'x_est': x_est, 'P_est': P_est}

    def update(self, poses):
        filtered_poses = {}

        for id, (rvec, tvec) in poses.items():
            if np.all(rvec == [[None, None, None]]) or np.all(tvec == [[None, None, None]]):
                filtered_poses[id] = ([[None, None, None]], [[None, None, None]])
                continue

            if id not in self.filters:
                self.init_filter(id)

            # Ensure rvec and tvec are numpy arrays and flatten them
            rvec = np.array(rvec).flatten()
            tvec = np.array(tvec).flatten()

            # Create the measurement vector
            z = np.hstack((rvec, tvec))

            # 预测
            x_pred = self.F @ self.filters[id]['x_est']
            P_pred = self.F @ self.filters[id]['P_est'] @ self.F.T + self.Q
            # print('########################################')
            # print(x_pred)
            # print(P_pred)
            # print(z)

            # 更新
            K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
            self.filters[id]['x_est'] = x_pred + K @ (z - self.H @ x_pred)
            self.filters[id]['P_est'] = (np.eye(12) - K @ self.H) @ P_pred

            filtered_rvec = self.filters[id]['x_est'][:3].reshape((1, 3))
            filtered_tvec = self.filters[id]['x_est'][6:9].reshape((1, 3))
            filtered_poses[id] = (filtered_rvec, filtered_tvec)

        return filtered_poses


def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


# (alpha, beta, gamma) = (roll, pitch, yaw) = (x, y, z)
def create_rotation_matrix(alpha, beta, gamma):
    r11 = np.cos(gamma)*np.cos(beta)
    r12 = -np.sin(gamma)*np.cos(alpha)+np.cos(gamma)*np.sin(beta)*np.sin(alpha)
    r13 = np.sin(gamma)*np.sin(alpha)+np.cos(gamma)*np.sin(beta)*np.cos(alpha)
    r21 = np.sin(gamma)*np.cos(beta)
    r22 = np.cos(gamma)*np.cos(alpha)+np.sin(gamma)*np.sin(beta)*np.sin(alpha)
    r23 = -np.cos(gamma)*np.sin(alpha)+np.sin(gamma)*np.sin(beta)*np.cos(alpha)
    r31 = -np.sin(beta)
    r32 = np.cos(beta)*np.sin(alpha)
    r33 = np.cos(beta)*np.cos(alpha)
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


def fit_circle_3d(points):
    # STEP 1 : Calculate the normal_fitted
    # Calculate the vector points
    v_array = np.empty((0, 3))
    for i in range(int(len(points) / 2)):
        vec_between_points = points[i] - points[int(len(points) / 2 + i)]
        v_array = np.append(v_array, np.array([vec_between_points]), axis=0)

    # Calculate fitted normal (rotation axis)
    n_array = np.empty((0, 3))
    for i in range(len(v_array) - 1):
        vec_normal = np.cross(v_array[i], v_array[i + 1])
        n_array = np.append(n_array, np.array([vec_normal]), axis=0)
    normal_fitted = np.mean(n_array, axis=0)
    normal_fitted = normal_fitted / np.linalg.norm(normal_fitted)  # + or - depends on camera position

    # STEP 2 : Project center on the rotation axis
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the initial radius as the distance from centroid to the first point
    radius_initial = np.mean(np.linalg.norm(points - centroid, axis=1))

    # Initial guess for the parameters
    center_initial = centroid

    initial_params = np.hstack((center_initial, radius_initial))

    # Perform least squares fitting
    def circular_residual_equation_1(params, points):
        center, radius = params[:3], params[3]
        residual = np.linalg.norm(points - center, axis=1) - radius
        return residual

    result = least_squares(circular_residual_equation_1, initial_params, args=(points,))

    # Extract the fitted parameters
    center_estimated = result.x[:3]
    # radius_estimated = result.x[3]

    # STEP 3 : Calculate center_fitted, radius_fitted
    def circular_residual_equation_2(t, points, C, v):
        # Calculate the point on the line C + t*v
        P = C + t * v
        # Calculate the distance from P to each point in points
        distances = np.linalg.norm(points - P, axis=1)
        # Return the mean distance
        return np.mean(distances)

    t_initial = 0.0

    # Perform least squares optimization to find the optimal t
    result = least_squares(circular_residual_equation_2, t_initial, args=(points, center_estimated, normal_fitted))

    # Calculate the optimal point
    t_optimal = result.x[0]
    center_fitted = center_estimated + t_optimal * normal_fitted
    radius_array = np.array([np.linalg.norm(point - center_fitted) for point in points])
    radius_fitted = np.mean(radius_array)

    return center_fitted, radius_fitted, normal_fitted


def project_point_to_plane(A, O, n):
    """Project the point A onto the plane with O as the centre and n as the normal vector."""
    n = n / np.linalg.norm(n)
    AO = A - O
    distance = np.dot(AO, n)
    P = A - distance * n
    return P


def find_closest_point_on_circle(A, O, R, n):
    """Find the point on circle C that is closest to point A."""
    P = project_point_to_plane(A, O, n)
    V = P - O
    V_unit = V / np.linalg.norm(V)
    B = O + R * V_unit
    return B


def angle_between_vectors(u, v, n):
    """Calculate the angle between vectors u and v around the normal vector n"""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    u_unit = u / np.linalg.norm(u)
    v_unit = v / np.linalg.norm(v)
    cos_theta = np.dot(u_unit, v_unit)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 确保值在[-1, 1]之间，避免数值误差
    # Set the direction of angle
    if np.dot(np.cross(u, v), n) < 0:
        angle = -angle
    return angle


def rotation_matrix_to_angle(R):
    """
    @Description: Convert rotation matrix to rotation angle (rotation around fixed axis)
    @Input: R - 3x3 rotation matrix
    @Output: theta - angle of rotation in radians
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    return theta


def rvec_to_quaternion(rvec):
    # 计算旋转角度 theta
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        # 如果角度非常小，直接返回单位四元数
        return np.array([1.0, 0.0, 0.0, 0.0])

    # 计算旋转轴 u
    u = rvec / theta

    # 计算四元数
    w = np.cos(theta / 2.0)
    x = u[0] * np.sin(theta / 2.0)
    y = u[1] * np.sin(theta / 2.0)
    z = u[2] * np.sin(theta / 2.0)

    return np.array([w, x, y, z])


def calculate_rotation_angle_from_quaternions(rvec1, rvec2):
    # 将旋转向量转换为四元数
    q1 = rvec_to_quaternion(rvec1)
    q2 = rvec_to_quaternion(rvec2)

    # 计算四元数的点积
    dot_product = np.dot(q1, q2)

    # 确保点积在 [-1, 1] 范围内
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 计算旋转角度
    delta_theta = 2 * np.arccos(dot_product)

    # To [-pi, pi]
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

    return delta_theta


def estimate_rotation_parameters(tag_positions):
    # tag_positions: List of tag positions over multiple frames
    all_points = np.concatenate(tag_positions, axis=0)
    center_fitted, radius_fitted, normal_fitted = fit_circle_3d(all_points)
    return center_fitted, radius_fitted, normal_fitted


def plot_circle(ax, center, radius, normal, num_points=100):
    # Creating a unit circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])

    # Constructing rotation matrices using orthogonal normal vectors
    normal = normal / np.linalg.norm(normal)
    if (normal == [0, 0, 1]).all():
        R = np.eye(3)
    else:
        v = np.cross([0, 0, 1], normal)
        c = np.dot([0, 0, 1], normal)
        s = np.linalg.norm(v)
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * ((1 - c) / (s ** 2))

    # Rotate and translate the circle
    circle_rot = R @ circle
    circle_rot[0, :] += center[0]
    circle_rot[1, :] += center[1]
    circle_rot[2, :] += center[2]

    # Drawing Circles
    ax.plot(circle_rot[0, :], circle_rot[1, :], circle_rot[2, :], 'r-')


