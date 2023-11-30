import cv2
import numpy as np

# Define the dimensions of the chessboard
chessboard_size = (7, 5)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,4,0)
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

calib_photos_path = "configs/calib_frames_pc"
calib_photos = [calib_photos_path + "/calib_frame" + str(i) + ".jpg" for i in range(1, 6)]

# Loop through your images
for filename in calib_photos:  # replace with your list of image filenames
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Now you can use camera_matrix and dist_coeffs for undistorting your images using cv2.undistort()
