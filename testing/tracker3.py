import cv2
import numpy as np
import time
import pandas as pd
from video_utils import saveVideo

#initialize result dataframe
result = pd.DataFrame(columns=['frame', 'x', 'y'])

def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.2

cap = cv2.VideoCapture('test_data/ret_masked.avi')

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)

#init kalman filter, first frame
init = False

#to save video
video_frames = []

while cap.isOpened():
    intensity_sum = 0
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    frame_diff = cv2.absdiff(gray, prev_gray)

    _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #take orizontal line from 600 to 680 pixels
    line = thresh[350:400, :]
    #sum all the pixels in the line
    intensity_sum = np.sum(line)
    #normalize the sum between 0 and 255
    intensity_sum = (intensity_sum/255)/80
    #drow a circle in the frame to show intensity sum, more intense make it whiter
    color = int(intensity_sum*255)
    cv2.circle(thresh, (100, 375), 40, (color, color, color), 2)

    #show the frame
    cv2.imshow('frame', thresh)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()

