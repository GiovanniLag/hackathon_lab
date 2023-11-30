import cv2
import numpy as np
import pandas as pd
import argparse
from video_utils import startLiveCamera
from fpdf import FPDF

def main(args):
    cname = args.cname
    # Create a Charuco board object
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((12, 8), 0.02, 0.015, dictionary)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    print(board.getChessboardSize())

    # Capture multiple images of the board from various angles and distances
    cap = startLiveCamera(0)
    images = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #take photo with spacebar
        if cv2.waitKey(1) & 0xFF == ord(' '):
            images.append(frame)
            print("photo taken")
        cv2.imshow('frame', frame)
        if len(images) == 5:
            break
    cap.release()

    # Lists to store the detected points
    all_charuco_corners = []
    all_charuco_ids = []

    # Loop through your images
    for image in images:
        # Convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(gray)
        print("Number of markers detected: ", len(corners))

        # If some markers are found, refine their positions and find the Charuco corners
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            print("Number of charuco corners detected: ", charuco_corners)


            # If enough corners are found and the number of ids matches the number of corners, add them to the list
            if charuco_ids is not None and charuco_corners is not None and len(charuco_ids) > 3 and charuco_ids.size == charuco_corners.shape[0]:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    print("Number of images processed: ", len(images))
    print("Number of charuco corners detected: ", len(all_charuco_corners))
    print("Number of charuco ids detected: ", len(all_charuco_ids))
    # Calibrate the camera using the detected points
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray.shape[::-1], None, None)

    # Save calibration info in "configs/cameras_calibration.csv"
    try:
        configs_data = pd.read_csv("configs/cameras_calibration.csv")
    except: #if cant find csv create it
        configs_data = pd.DataFrame({'camera_name':cname, 'ret':ret, 'camera_matrix':camera_matrix, 'dist_coeffs':dist_coeffs, 'rvecs':rvecs, 'tvecs':tvecs}, index=None)
        
    #save camera_name, calibration parameters
    pd.concat([configs_data, pd.DataFrame({'camera_name':cname, 'ret':ret, 'camera_matrix':camera_matrix, 'dist_coeffs':dist_coeffs, 'rvecs':rvecs, 'tvecs':tvecs}, index=None)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera')
    parser.add_argument('--cname', type=str, default='camera1', help='camera name')
    args = parser.parse_args()
    main(args)