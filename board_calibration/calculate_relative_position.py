import cv2
import matplotlib
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def rotation_matrix_to_euler_angles(R):
    """
    @Description: Convert a rotation matrix to Euler angles (yaw, pitch, roll).
    @Input: R - 3x3 rotation matrix
    @Output: (yaw, pitch, roll) - Euler angles
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arcsin(-R[2, 0])
        roll = 0

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def rotation_matrix_to_angle(R):
    """
    @Description: Convert rotation matrix to rotation angle (rotation around fixed axis)
    @Input: R - 3x3 rotation matrix
    @Output: theta - angle of rotation in radians
    """
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)
    return np.degrees(theta)


def list_files_in_directory(directory_path):
    try:
        items = os.listdir(directory_path)

        # Filter out all folders
        files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]

        return files
    except FileNotFoundError:
        return "directory_path does not exist"
    except Exception as e:
        return f"Error: {e}"


# Choose whether to use ChArUco Boards or chessboard for detection
use_charuco = False  # Set this variable to True to use ChArUco Boards, False to use chessboard
if use_charuco:
    print("Using ChArUco Boards.")
else:
    print("Using Chess Boards.")

# Prepare calibration plate parameters
if use_charuco:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(6, 4, 0.026, 0.013, aruco_dict)
else:
    pattern_size = (9, 6)  # Number of grids for calibration plate
    square_size = 2.1  # Actual size of each grid (can be centimetres)
    # Generate the coordinates of the corner points of the calibration plate in the world coordinate system.
    object_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    object_points *= square_size

dataset_name = '2'
testset_name = '3'

# Read camera calibration data
if use_charuco:
    calibration_path = './images/camera_calibration/charuco_board/calibration_dataset_' + dataset_name + '/'
else:
    calibration_path = './images/camera_calibration/chess_board/calibration_dataset_' + dataset_name + '/'
with open(calibration_path + "calibration_results/" + "camera_calibration_data.pkl", 'rb') as f:
    data = pickle.load(f)

camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

if use_charuco:
    test_path = './images/camera_calibration/charuco_board/calibration_testset_' + testset_name + '/'
else:
    test_path = './images/camera_calibration/chess_board/calibration_testset_' + testset_name + '/'

test_files = list_files_in_directory(test_path)

theta_list = np.array([])

for cnt in range(len(test_files)-1):
    # Calculate pose from corners_img1
    img1 = cv2.imread(test_path + "cal_test_" + str(cnt) + ".png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if use_charuco:
        # Detect ChArUco corners
        corners_img1, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img1, aruco_dict)
        if len(corners_img1) > 0:
            found1, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners_img1, ids, img1, board)
            if found1:
                retval1, rvec1, tvec1 = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
                                                                           camera_matrix, dist_coeffs)

    else:
        # Detect chessboard corners
        found1, corners_img1 = cv2.findChessboardCorners(img1, pattern_size)
        if found1:
            retval1, rvec1, tvec1 = cv2.solvePnP(object_points, corners_img1, camera_matrix, dist_coeffs)

    # Calculate pose from corners_img1
    img2 = cv2.imread(test_path + "cal_test_" + str(cnt+1) + ".png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if use_charuco:
        # Detect ChArUco corners
        corners_img2, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img2, aruco_dict)
        if len(corners_img2) > 0:
            found2, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners_img2, ids, img2, board)
            if found2:
                retval2, rvec2, tvec2 = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
                                                                           camera_matrix, dist_coeffs)
    else:
        # Detect chessboard corners
        found2, corners_img2 = cv2.findChessboardCorners(img2, pattern_size)
        if found2:
            retval2, rvec2, tvec2 = cv2.solvePnP(object_points, corners_img2, camera_matrix, dist_coeffs)

    if found1 and found2:
        # Convert a rotation vector to a rotation matrix
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)

        # Calculate relative rotation matrix
        R_rel = R2 @ R1.T

        # Calculate relative translation vector
        t_rel = tvec2 - R_rel @ tvec1

        # print("Relative rotation matrix:")
        # print(R_rel)
        # print("Relative translation vector:")
        # print(t_rel)

        yaw, pitch, roll = rotation_matrix_to_euler_angles(R_rel)
        print(f"Yaw: {yaw:.2f} degrees", f"Pitch: {pitch:.2f} degrees", f"Roll: {roll:.2f} degrees")

        theta = rotation_matrix_to_angle(R_rel)
        print(f"Theta: {theta:.2f} degrees")
        theta_list = np.append(theta_list, theta)
    else:
        print("Cannot calculate rotation angle between <cal_test_" + str(cnt) + ".png> and <cal_test_" +
              str(cnt+1) + ".png>")

mean_theta = np.mean(theta_list)
max_theta = np.max(theta_list)
min_theta = np.min(theta_list)

plt.figure(figsize=(10, 6))
plt.plot(theta_list, label='Theta Values')

plt.axhline(y=mean_theta, color='r', linestyle='--', label=f'Mean: {mean_theta:.2f}')
plt.axhline(y=max_theta, color='g', linestyle='-.', label=f'Max: {max_theta:.2f}')
plt.axhline(y=min_theta, color='b', linestyle=':', label=f'Min: {min_theta:.2f}')

plt.ylim([0, 15])
plt.legend()
plt.title('Theta List with Reference Lines')
plt.xlabel('Index')
plt.ylabel('Theta Value')
plt.show()
