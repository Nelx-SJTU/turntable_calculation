import cv2
import os
import numpy as np
import pickle


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

dataset_name = '4'

# Get calibration images
if use_charuco:
    calibration_path = './images/camera_calibration/charuco_board/calibration_dataset_' + dataset_name + '/'
else:
    calibration_path = './images/camera_calibration/chess_board/calibration_dataset_' + dataset_name + '/'
image_files = list_files_in_directory(calibration_path)

print("Camera calibration dataset path:", calibration_path)
print("Camera calibration images:", image_files)

# Used to store the coordinates of the corner points in all images
object_points_list = []
image_points_list = []

found_corner_cnt = 0

if use_charuco:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(6, 4, 0.026, 0.013, aruco_dict)

# Reads images and detects corner points
for image_file in image_files:
    image = cv2.imread(calibration_path + image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_charuco:
        # Detect ChArUco corners
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            found, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if found:
                object_points_list.append(board)
                image_points_list.append(charucoCorners)
                found_corner_cnt += 1
                cv2.aruco.drawDetectedCornersCharuco(image, charucoCorners, charucoIds)
                cv2.imshow('Corners', image)
                cv2.waitKey(300)
    else:
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                   cv2.CALIB_CB_FAST_CHECK)
        print("Processing " + calibration_path + image_file, "Found_corner = ", found)
        if found:
            image_points_list.append(corners)
            object_points_list.append(object_points)
            cv2.drawChessboardCorners(image, pattern_size, corners, found)
            found_corner_cnt += 1
            cv2.imshow('Corners', image)
            cv2.waitKey(300)

cv2.destroyAllWindows()

# Calibration the camera
if found_corner_cnt >= 15:
    found, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, gray.shape[::-1], None, None)
    print("In-camera parameter matrix: ")
    print(camera_matrix)
    print("distortion factor:")
    print(dist_coeffs)

    # Save results
    if not os.path.exists(calibration_path + "calibration_results"):
        os.makedirs(calibration_path + "calibration_results")

    with open(calibration_path + "calibration_results/" + "camera_calibration_data.pkl", 'wb') as f:
        pickle.dump({
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
        }, f)

    print("Data is saved to" + calibration_path + "calibration_results/" + "camera_calibration_data.pkl")

else:
    print("There are not enough images which finds the corners. ")
    print("Camera calibration failed.")
