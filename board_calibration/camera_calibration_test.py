import cv2
import pickle
import numpy as np


# Choose whether to use ChArUco Boards or chessboard for detection
use_charuco = False  # Set this variable to True to use ChArUco Boards, False to use chessboard
if use_charuco:
    print('Using ChArUco Boards.')
else:
    print('Using Chess Boards.')

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

# Read camera calibration data
if use_charuco:
    calibration_path = './images/camera_calibration/charuco_board/calibration_dataset_' + dataset_name + '/'
else:
    calibration_path = './images/camera_calibration/chess_board/calibration_dataset_' + dataset_name + '/'
with open(calibration_path + "calibration_results/" + "camera_calibration_data.pkl", 'rb') as f:
    data = pickle.load(f)

camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

print("In-camera parameter matrix: ")
print(camera_matrix)
print("distortion factor:")
print(dist_coeffs)

testset_name = '3'
# Reads a new image containing the calibration plate
if use_charuco:
    test_img_path = './images/camera_calibration/charuco_board/calibration_testset_' + testset_name + '/cal_test_10.png'
else:
    test_img_path = './images/camera_calibration/chess_board/calibration_testset_' + testset_name + '/cal_test_10.png'
new_image = cv2.imread(test_img_path)
gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

if use_charuco:
    # Detect ChArUco corners
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        found, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if found:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs)

else:
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        # Calculate the pose
        retval, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

# Display results
if found:
    # Rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    print("Rotation Matrix:\n", rotation_matrix)
    print("Translation Vector:\n", tvec)
else:
    print("Calibration pattern not found in the image.")
