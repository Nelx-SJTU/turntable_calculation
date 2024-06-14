import cv2
import pickle
import numpy as np


cap = cv2.VideoCapture(2)

# Set camera resolution
desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# 加载相机内参和畸变系数
dataset_name = '3'
calibration_path = './images/camera_calibration/chess_board/calibration_dataset_' + dataset_name + '/'
with open(calibration_path + "calibration_results/" + "camera_calibration_data.pkl", 'rb') as f:
	data = pickle.load(f)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
print('Camera parameters path: ' + calibration_path)

# 设置ArUco字典和检测参数
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters_create()

marker_size = 1.95
axis_length = 0.1

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
	cv2.imshow("USB", frame)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('c'):
		img = frame
		corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

		if ids is not None:
			rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

			cv2.aruco.drawDetectedMarkers(img, corners, ids)

			axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)

			for i in range(len(ids)):
				# 获取旋转向量和平移向量
				rvec = rvecs[i][0]
				tvec = tvecs[i][0]

				# 绘制坐标轴
				cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

				# 打印旋转向量和平移向量
				print(f"ID: {ids[i]}")
				print(f"Rotation Vector (rvec): {rvec}")
				print(f"Translation Vector (tvec): {tvec}")

		cv2.imshow('Detected markers', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
