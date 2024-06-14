import matplotlib
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from calibration_utils import *

matplotlib.use('TkAgg')

cap = cv2.VideoCapture(2)

# Set camera resolution
desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Read the current resolution of the camera
current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Current resolution: {current_width}x{current_height}")

print('Visual calibration mode: Aruco Tags')

# Prepare calibration plate parameters
# Define Aruco dictionary
aruco_dict_type = cv2.aruco.DICT_5X5_1000
aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
# Initialization detector's parameters
parameters = cv2.aruco.DetectorParameters_create()
marker_size = 1.95  # length of marker's sides (in cm)

# Read camera calibration data
dataset_name = '4'
calibration_path = './images/camera_calibration/chess_board/calibration_dataset_' + dataset_name + '/'
# Load camera calibration parameters
with open(calibration_path + "calibration_results/" + "camera_calibration_data.pkl", 'rb') as f:
	data = pickle.load(f)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
print('Camera parameters path: ' + calibration_path)

draw_position = False
draw_angle = False

# Plot settings
if draw_position:
	rotation_fig = plt.figure()
	camera_center_ax = rotation_fig.add_subplot(121, projection='3d')
	camera_center_ax.set_xlabel('X')
	camera_center_ax.set_ylabel('Y')
	camera_center_ax.set_zlabel('Z')

	turntable_center_ax = rotation_fig.add_subplot(122, projection='3d')
	turntable_center_ax.set_xlabel('X')
	turntable_center_ax.set_ylabel('Y')
	turntable_center_ax.set_zlabel('Z')

if draw_angle:
	angle_fig, angle_time_ax = plt.subplots()
	angle_time_ax.set_xlabel('Time')
	angle_time_ax.set_ylabel('Angle')

# Coordinate of camera
camera_coordinate_c = np.array([0, 0, 0])

# Image capture interval (seconds)
detection_interval = 0.1
calculation_interval = 0.3

# Minimum number of images for trajectory fitting
trajectory_fitting_num_min = 20

# Aruco tag list  [93, 102, 98, 103, 105, 97, 107, 95, 96, 104, 94, 106]
aruco_tags_id_list = np.array([94, 101, 92, 104, 93, 100, 87, 96, 102, 90, 86, 95,
							   98, 99, 107, 103, 91, 89, 97, 105, 85, 88, 106])

aruco_detector = ArucoDetector(aruco_dict_type=aruco_dict_type,
							   aruco_tags_id_list=aruco_tags_id_list,
							   marker_size=marker_size,
							   camera_matrix=camera_matrix,
							   dist_coeffs=dist_coeffs,
							   draw=True)

# To make rvec more dependent on the sensor value, we can increase the value of Qrvec.
# To make rvec more dependent on the sensor value, we can decrease the value of Rrvec.
# Default value: q: 0.01, r: 0.1
kalman_filter_aruco = KalmanFilterAruco(dt=detection_interval,
										q_rvec=0.8, q_tvec=0.018,
										r_rvec=0.001, r_tvec=0.05)

track_len = 20
plot_len = 1
turntable_aruco_tag_set = ArucoTagSet(set_name='Turntable Aruco Tags Set',
									  track_len=track_len,
									  aruco_tags_id_list=aruco_tags_id_list,
									  aruco_dict_type=aruco_dict_type)

turntable_diameter = 18.0
turntable_trajectory_track_len = 5
turn_table = TurnTable(aruco_tags_id_list=aruco_tags_id_list,
					   turntable_diameter=turntable_diameter,
					   camera_coordinate_c=camera_coordinate_c,
					   trajectory_track_len=turntable_trajectory_track_len)

# tag_positions_t = FixedSizeArray(max_len=10, data_size=(0, len(aruco_tags_id_list), 3), axis=0)
# tag_positions_t = []
# for i in range(len(aruco_tags_id_list)):
# 	tag_positions_t.append(FixedSizeArray(max_len=15, data_size=(0, 3), axis=0))

cv2.namedWindow('USB', 0)


start_calibration = False
finish_calibration = False
zero_reference_set = False
print_info = False

frame_cnt = 0

detection_start_time = time.time()
calculation_start_time = time.time()

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
	cv2.imshow("USB", frame)

	k = cv2.waitKey(1) & 0xFF
	if k == ord('c'):
		start_calibration = True
		print("Start collecting calibration images.")
	elif k ==ord('o'):
		# Reset zero degree
		print('Start to rest zero degree reference.')
		# zero_reference_set = turn_table.set_zero_reference_with_single_frame()
		zero_reference_set = turn_table.set_zero_reference_with_multiple_frames()
		current_angle_filtered = 0.0
		current_angle_filtered_list = []
		set_ref_0_time = time.time()
	elif k == ord('q'):
		break

	if start_calibration and time.time() - detection_start_time >= detection_interval:
		detection_start_time = time.time()

		# Get the camera frame
		img = np.array(frame)

		# Detect Aruco corners
		aruco_tag_poses, found_all = aruco_detector.detect_aruco_poses(image=img)

		# Kalman filter
		aruco_tag_poses_filtered = kalman_filter_aruco.update(aruco_tag_poses)

		# Update the Aruco tag set
		turntable_aruco_tag_set.update(poses=aruco_tag_poses_filtered)

		if print_info:
			turntable_aruco_tag_set.print_info()

		if not found_all:
			print("Unable to locate all aruco tags in 'aruco_tags_id_list'.")

		if time.time() - calculation_start_time > calculation_interval:
			calculation_start_time = time.time()
			frame_cnt += 1

			# Check whether the number of images is enough for trajectory fitting
			if not finish_calibration and frame_cnt >= trajectory_fitting_num_min:
				print('Start calculating calibration parameters.')

				# Fitting the rotation around fixed axis
				turn_table.circular_rotation_fitting(aruco_tag_set=turntable_aruco_tag_set)
				finish_calibration = True

				# Transform the camera from camera-centered coordinate to turntable-centered coordinate
				turn_table.camera_coordinate_transform()

				# Plot the calibration result
				if draw_position:
					camera_center_ax.cla()
					turntable_center_ax.cla()
					turn_table.plot_calibration_result(camera_center_ax=camera_center_ax)

			if finish_calibration:
				for id in aruco_tags_id_list:
					turn_table.update(id=id,
									  rotation_data=turntable_aruco_tag_set.get_data()[id].get_data().get_dict()['rv_c'][-1],
									  translation_data=turntable_aruco_tag_set.get_data()[id].get_data().get_dict()['tv_c'][-1])

				if draw_position:
					turn_table.plot_points(ax=turntable_center_ax, plot_length=plot_len)

				if zero_reference_set:
					currect_position_on_circle, current_angle = turn_table.calculate_rotation_angle()
					print(f"Current angle: {np.degrees(current_angle):.2f} degrees")
					
					# Filter angle
					if len(current_angle_filtered_list) > 6:
						current_angle_filtered_list.pop(0)
						current_angle_filtered_list.append(current_angle)
					else:
						current_angle_filtered_list.append(current_angle)
					# current_angle_filtered = 0.6*current_angle_filtered + 0.4*current_angle
					current_angle_filtered = np.mean(current_angle_filtered_list)
					print(f"Current angle filtered: {np.degrees(current_angle_filtered):.2f} degrees")

					if draw_angle:
						angle_time_ax.scatter(time.time() - set_ref_0_time, current_angle_filtered, 
							color='red', marker='o', label='Angle')
						angle_time_ax.set_xlim([time.time() - set_ref_0_time - 20, time.time() - set_ref_0_time])
					
					if draw_position:
						turn_table.plot_rotation(ax=turntable_center_ax,
													currect_position_on_circle=currect_position_on_circle)

			plt.pause(0.01)

cap.release()
cv2.destroyAllWindows()
