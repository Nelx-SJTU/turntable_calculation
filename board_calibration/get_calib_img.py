import cv2
import os


def get_camera_resolutions(cap):
	# Common resolution lists
	common_resolutions = [
		(1920, 1080),  # Full HD
		(1280, 720),  # HD
		(640, 480),  # VGA
		(320, 240),  # QVGA
	]
	available_resolutions = []

	# Check whether the camera support these resolutions
	for width, height in common_resolutions:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		if actual_width == width and actual_height == height:
			available_resolutions.append((actual_width, actual_height))

	return available_resolutions


cap = cv2.VideoCapture(0)

# Read the current resolution of the camera
current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Current resolution: {current_width}x{current_height}")

# Getting the resolution available to the camera
available_resolutions = get_camera_resolutions(cap)
print("Available resolutions:")
for res in available_resolutions:
	print(f"{res[0]}x{res[1]}")

# Set camera resolution
desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Check if the settings are successful
new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Current resolution: {new_width}x{new_height}")


index = 0

get_dataset = True  # True for getting dataset, False for getting test set

# Choose whether to use ChArUco Boards or chessboard for detection
use_charuco = True  # Set this variable to True to use ChArUco Boards, False to use chessboard
if use_charuco:
	print('Using ChArUco Boards.')
else:
	print('Using Chess Boards.')

set_name = '2'

if get_dataset:
	if use_charuco:
		save_path = './images/camera_calibration/charuco_board/calibration_dataset_' + set_name + '/'
	else:
		save_path = './images/camera_calibration/chess_board/calibration_dataset_' + set_name + '/'
else:
	if use_charuco:
		save_path = './images/camera_calibration/charuco_board/calibration_testset_' + set_name + '/'
	else:
		save_path = './images/camera_calibration/chess_board/calibration_testset_' + set_name + '/'

if not os.path.exists(save_path):
	os.makedirs(save_path)

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
	cv2.imshow("USB", frame)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('s'):
		if get_dataset:
			cv2.imwrite(save_path + "cal_" + str(index) + ".png", frame)
			print("Save calib data img :" + str(index))
		else:
			cv2.imwrite(save_path + "cal_test_" + str(index) + ".png", frame)
			print("Save calib test img :" + str(index))
		index += 1
	elif k == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
