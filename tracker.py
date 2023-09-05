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
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    frame_diff = cv2.absdiff(gray, prev_gray)

    _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        centroid = get_centroid(c)
        #add circle for centroid in frame
        cv2.circle(frame, centroid, 10, (0, 0, 255), 2)

        if not init:
            kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], np.float32)
            init = True
        
        kalman.correct(np.array([np.float32(centroid[0]), np.float32(centroid[1])]))
        predicted = kalman.predict()
        
        if centroid != (0, 0):
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 10, (0, 255, 255), 2)
            result = pd.concat([result, pd.DataFrame({'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 'x': int(predicted[0]), 'y': int(predicted[1])}, index=[0])], ignore_index=True)

    cv2.imshow('Frame', frame)
    video_frames.append(frame)
    #time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()

#save result
result.to_csv('test_results/result_1.csv', index=False)

#save video
saveVideo(video_frames, 'test_data/results/result_1.avi', fps=29)
