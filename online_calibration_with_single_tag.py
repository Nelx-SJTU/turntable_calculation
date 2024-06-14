import cv2
import matplotlib
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from calibration_utils import *

matplotlib.use('TkAgg')

cap = cv2.VideoCapture(2)

# Set camera resolution
# desired_width = 1280
# desired_height = 720
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Read the current resolution of the camera
current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Current resolution: {current_width}x{current_height}")

# Choose whether to use Aruco Tags or Chessboard for detection
use_aruco = True  # Set this variable to True to use Aruco Tags, False to use Chessboard
if use_aruco:
	print('Visual calibration mode: Aruco Tags')
else:
	print('Visual calibration mode: Chess Boards')

# Prepare calibration plate parameters
if use_aruco:
	# Define Aruco dictionary
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
	# Initialization detector's parameters
	parameters = cv2.aruco.DetectorParameters_create()
	marker_size = 5.8  # length of marker's sides (in cm)
else:
	pattern_size = (9, 6)  # Number of grids for calibration plate
	square_size = 2.1  # Actual size of each grid (can be centimetres)
	# Generate the coordinates of the corner points of the calibration plate in the world coordinate system.
	object_points = np.zeros((np.prod(pattern_size), 3), np.float32)
	object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
	object_points *= square_size

# Read camera calibration data
dataset_name = '2'
calibration_path = './images/camera_calibration/chess_board/calibration_dataset_' + dataset_name + '/'
# Load camera calibration parameters
with open(calibration_path + "calibration_results/" + "camera_calibration_data.pkl", 'rb') as f:
	data = pickle.load(f)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
print('Camera parameters path: ' + calibration_path)

# Plot settings
fig = plt.figure()
camera_center_ax = fig.add_subplot(211, projection='3d')
camera_center_ax.set_xlabel('X')
camera_center_ax.set_ylabel('Y')
camera_center_ax.set_zlabel('Z')
camera_coordinate_c = np.array([0, 0, 0])

turntable_center_ax = fig.add_subplot(212, projection='3d')
turntable_center_ax.set_xlabel('X')
turntable_center_ax.set_ylabel('Y')
turntable_center_ax.set_zlabel('Z')

# Use fix size array to note the board's pose
rotation_vectors = FixedSizeArray(max_len=15, data_size=(0, 3), axis=0)
translation_vectors = FixedSizeArray(max_len=15, data_size=(0, 3), axis=0)
chessboard_positions_t = FixedSizeArray(max_len=15, data_size=(0, 3), axis=0)

# Image capture interval (seconds)
capture_interval = 0.5

# Minimum number of images for trajectory fitting
trajectory_fitting_num_min = 10

# Aruco tag list
aruco_tag_list = np.array([[48], [624], [684]])
aruco_tag_list = np.array([[684]])

start_time = time.time()
start_calibration = False
finish_calibration = False
found = False

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
	cv2.imshow("USB Camera", frame)

	k = cv2.waitKey(1) & 0xFF
	if k == ord('c'):
		start_calibration = True
		print("Start calibration.")
	# if k == ord('f'):
	# 	finish_calibration = True
	# 	print("Finish calibration.")
	elif k == ord('q'):
		break

	if start_calibration and time.time() - start_time >= capture_interval:
		start_time = time.time()
		# image_array.append(np.array(frame))
		img = np.array(frame)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if use_aruco:
			# Detect Aruco corners
			corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
			if ids is not None:
				found = True
				rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
				rvec = rvecs[np.where(ids == aruco_tag_list[0])]
				tvec = tvecs[np.where(ids == aruco_tag_list[0])]
			else:
				found = False
		else:
			# Detect chessboard corners
			found, corners = cv2.findChessboardCorners(img, pattern_size)
			if found:
				retval, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

		# Fill the fix size array
		if found:
			rotation_vectors.append(np.reshape(rvec, (1, 3)))
			translation_vectors.append(np.reshape(tvec, (1, 3)))
		else:
			print("Unable to locate feature points on the calibration plate. ")

		# Check whether the number of images is enough for trajectory fitting
		if not finish_calibration and len(rotation_vectors.data) >= trajectory_fitting_num_min:
			# Convert a rotation vector to a rotation matrix
			rotation_matrices = np.array([cv2.Rodrigues(rvec)[0] for rvec in rotation_vectors.data])  # Shape = (n,3,3)

			# Create a list of transformation matrices
			transformation_matrices = [create_transformation_matrix(rotation_matrices[i], translation_vectors.data[i])
									   for i in range(len(rotation_matrices))]

			# Get the position of the calibration plate in the world coordinate system.
			chessboard_positions_c = [transformation_matrices[i][:3, 3] for i in range(len(transformation_matrices))]

			# Least Squares Fitting of Circles
			fitted_center, fitted_radius, rotation_axis = fit_circle_3d(chessboard_positions_c)

			# Finish calibration
			print('Circular rotation fitting complete.')
			print(f"Fitted center: {fitted_center}")
			print(f"Fitted radius: {fitted_radius}")
			print(f"Rotation axis: {rotation_axis}")
			print("Calibration succeed. ")
			finish_calibration = True

			# Calculate coordinate transform parameters (Change to turntable centered coordinate)
			camera_translation_vector = camera_coordinate_c - fitted_center

			# Calculate new base of x-axis (in turntable coordinate)
			new_x_axis_base = camera_translation_vector - np.dot(camera_translation_vector,
																 rotation_axis) * rotation_axis
			new_x_axis_base = new_x_axis_base / np.linalg.norm(new_x_axis_base)

			# Calculate new base of y-axis (in turntable coordinate)
			new_y_axis_base = np.cross(rotation_axis, new_x_axis_base)
			new_y_axis_base = new_y_axis_base / np.linalg.norm(new_y_axis_base)

			# Calculate the rotation matrix
			camera_to_turntable_R = np.array([new_x_axis_base, new_y_axis_base, rotation_axis]).T

			# Calculate new camera coordinate
			camera_coordinate_t = np.dot(camera_to_turntable_R.T, (camera_coordinate_c - fitted_center))
			# Calculate coordinate transform parameters finished.

			# Plot the calibration result
			camera_center_ax.cla()

			# Plot the camera coordinate to (0,0,0)
			camera_center_ax.scatter(camera_coordinate_c[0], camera_coordinate_c[1], camera_coordinate_c[2],
									 color='red', marker='o', label='Camera Origin')

			# Plot the fitted rotation center
			camera_center_ax.scatter(fitted_center[0], fitted_center[1], fitted_center[2], color='green', marker='x',
									 label='Rotation Center')

			# Plot the predicted circle
			plot_circle(camera_center_ax, fitted_center, fitted_radius, rotation_axis)

			# Plot the rotation axis
			normal_line = np.array([fitted_center - rotation_axis * fitted_radius,
									fitted_center + rotation_axis * fitted_radius])
			camera_center_ax.plot(normal_line[:, 0], normal_line[:, 1], normal_line[:, 2], color='green',
								  label='Rotation Axis (Normal Vector)')

			# Plot the position of calibration plate
			chessboard_positions_c = np.array(
				[transformation_matrices[i][:3, 3] for i in range(len(transformation_matrices))])
			camera_center_ax.scatter(chessboard_positions_c[:, 0], chessboard_positions_c[:, 1],
									 chessboard_positions_c[:, 2],
									 color='blue', label='Calibrationboard')

			# Set the XYZ axis scale the same
			# ax.set_xlim([-60, 60])
			# ax.set_ylim([-60, 60])
			# ax.set_zlim([-60, 60])
			camera_center_ax.axis('equal')

		# plt.show()

		if finish_calibration:
			# Convert a rotation vector to a rotation matrix
			rotation_matrices = np.array([cv2.Rodrigues(rvec)[0] for rvec in rotation_vectors.data])  # Shape = (n,3,3)

			# Create a list of transformation matrices
			transformation_matrices = [create_transformation_matrix(rotation_matrices[i], translation_vectors.data[i])
									   for i in range(len(rotation_matrices))]

			# Clear plot
			turntable_center_ax.cla()

			chessboard_positions_c = np.array(
				[transformation_matrices[i][:3, 3] for i in range(len(transformation_matrices))])
			for point in chessboard_positions_c:
				point_t = np.dot(camera_to_turntable_R.T, (point - fitted_center))
				# print("point_t", point_t)
				chessboard_positions_t.append(np.reshape(point_t[:3], (1, 3)))

			# Plot the camera coordinate to (0,0,0)
			turntable_center_ax.scatter(camera_coordinate_t[0], camera_coordinate_t[1], camera_coordinate_t[2],
										color='red', marker='o', label='Camera Origin')

			# Plot the fitted rotation center
			turntable_center_ax.scatter(0, 0, 0, color='green', marker='x', label='Rotation Center')

			# Plot the predicted circle
			plot_circle(turntable_center_ax, np.array([0, 0, 0]), fitted_radius, np.array([0, 0, 1]))

			# Plot the rotation axis
			normal_line = np.array([np.array([0, 0, 0]) - np.array([0, 0, 1]) * fitted_radius,
									np.array([0, 0, 0]) + np.array([0, 0, 1]) * fitted_radius])
			turntable_center_ax.plot(normal_line[:, 0], normal_line[:, 1], normal_line[:, 2], color='green',
									 label='Rotation Axis (Normal Vector)')

			# Plot the position of calibration plate
			turntable_center_ax.scatter(chessboard_positions_t.data[:, 0], chessboard_positions_t.data[:, 1],
										chessboard_positions_t.data[:, 2],
										color='blue', label='Calibrationboard')

			turntable_center_ax.scatter(chessboard_positions_t.data[-1, 0], chessboard_positions_t.data[-1, 1],
										chessboard_positions_t.data[-1, 2],
										color='blue', s=100, label='Calibrationboard')

			# Set the XYZ axis scale the same
			# ax.set_xlim([-60, 60])
			# ax.set_ylim([-60, 60])
			# ax.set_zlim([-60, 60])
			turntable_center_ax.axis('equal')

			plt.pause(0.01)
		# plt.show()

cap.release()
cv2.destroyAllWindows()
