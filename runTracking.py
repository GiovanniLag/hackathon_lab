import cv2
import numpy as np
import time
import pandas as pd
from video_utils import saveVideo, startLiveCamera
import argparse
from matplotlib import pyplot as plt

def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

def get_click_positions(frame):
    clicks_x = []
    clicks_y = []

    def click_event(event, x, y, flags, param):
        nonlocal clicks_x, clicks_y
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks_x.append(x)
            clicks_y.append(y)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow('Frame', frame)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    aspect_ratio = frame.shape[1] / frame.shape[0]
    window_width = 1080
    window_height = int(window_width / aspect_ratio)
    cv2.resizeWindow('Frame', window_width, window_height)
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('d'):
            cv2.destroyAllWindows()
            return clicks_x, clicks_y

def fit_distortion_profile(front, back, points_dist): #y = ax + b
    coeff = np.polyfit(front, back, 1)
    a = coeff[0]
    b = coeff[1]
    return a, b

def solve_distortion(ypx, a, b):
    diff = ypx * a - b
    return ypx + diff/2 #assume ball is in the middle of tube


def make_distortion_profile(frame):
    left_clicks = []
    right_clicks = []

    def click_event(event, x, y, flags, param):
        nonlocal left_clicks, right_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            left_clicks.append(y)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('Frame', frame)
        elif event == cv2.EVENT_RBUTTONDOWN:
            right_clicks.append(y)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('Frame', frame)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    aspect_ratio = frame.shape[1] / frame.shape[0]
    window_width = 1080
    window_height = int(window_width / aspect_ratio)
    cv2.resizeWindow('Frame', window_width, window_height)
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

    cv2.destroyAllWindows()
    return left_clicks, right_clicks #left is front right is back


def main(arg): #args are: 0 for live camera, 1 for video file and second arg is the path to the video file, third arg is the path to the output video file (second and third are optional)
    #initialize result dataframe
    result = pd.DataFrame(columns=['frame', 'x', 'y', 'time'])
    positions = pd.DataFrame(columns=['y', 'time'])

    csv_out = arg.output
    pos_csv_out = arg.output[:-4] + "_positions.csv"

    #set up the plot (there are some problem with the real time plot that makes the program crash after a while, so it's commented out for now)
    #plt.ion() 
    #fig, ax = plt.subplots()
    #flip y axis
    #ax.invert_yaxis()

    #initialize video stream
    if arg.source == 0:
        cap = startLiveCamera(0)
    elif arg.source == 1:
        cap = cv2.VideoCapture(arg.video_input)
    else:
        print("Invalid source")
        return

    #get first frame to have user config distance
    metric_points_x = []
    metric_points_y = []
    metric_points_x, metric_points_y = get_click_positions(cap.read()[1])
    points_dist = float(input("Enter distance between points in cm: "))
    points_dist = points_dist/100
    #pixel to length ratio
    px_to_len = points_dist/np.sqrt((metric_points_x[0]-metric_points_x[1])**2 + (metric_points_y[0]-metric_points_y[1])**2)

    # New distortion profile if specified
    if arg.dist_profile == "new":
        front,back = make_distortion_profile(cap.read()[1])
        #save to csv
        dist_profile = pd.DataFrame({'front_pos': front, 'back_pos': back})
        dist_profile.to_csv("configs/" + arg.output[:-4] + "_dist_profile.csv", index=False)
    # Distortion profile (read from file)
    else:
        print("Using distortion profile from file")
        if arg.dist_profile == None:
            path_to_distprof = "configs/dist_profile.csv"
        else:
            path_to_distprof = arg.dist_profile
        dist_profile = pd.read_csv(path_to_distprof)
        front = dist_profile['front_pos'].values
        back = dist_profile['back_pos'].values
    
    a,b = fit_distortion_profile(front, back, points_dist)

    #now we proceed with tracking
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.2

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)

    #init kalman filter, first frame
    init = False

    #to save video
    #video_frames = []

    while cap.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("quit")
            cv2.destroyAllWindows()
            cap.release()
            break

        ret, frame = cap.read()
        frame_time = time.time()
        if not ret:
            break

        # Apply filters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
        gray = cv2.GaussianBlur(gray, (11, 11), 0) #apply gaussian blur

        #get difference between frames
        frame_diff = cv2.absdiff(gray, prev_gray) 

        _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY) #threshold the difference

        #apply morphological opening
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        #find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            centroid = get_centroid(c)
            #add circle for centroid in frame
            cv2.circle(frame, centroid, 10, (0, 0, 255), 2)

            if not init:
                kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], np.float32)
                init = True
                first_frame_time = frame_time

            kalman.correct(np.array([np.float32(centroid[0]), np.float32(centroid[1])]))
            predicted = kalman.predict()

            if centroid != (0, 0):
                cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 10, (0, 255, 255), 2)
                result = pd.concat([result, pd.DataFrame({'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 'x': int(predicted[0]), 'y': int(predicted[1]), 'time':frame_time}, index=[0])], ignore_index=True)
                predicted_y = int(predicted[1])
                #solve distortion
                predicted_y = solve_distortion(predicted_y, a, b)
                #convert to metric
                predicted_y = predicted_y*px_to_len
                #save to positions
                positions = pd.concat([positions, pd.DataFrame({'y': predicted_y, 'time':frame_time-first_frame_time}, index=[0])], ignore_index=True)

        cv2.imshow('Frame', frame)
        #To save video frames (commented out for now)
        #video_frames.append(frame)

        #Real time plotting
        #ax.plot(positions['time'], positions['y'])
        #ax.set(xlabel='time (s)', ylabel='y (m)', title='y vs time')
        #ax.grid()
        #fig.canvas.draw()
        #fig.canvas.flush_events()

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

    #save the csv file
    result.to_csv(csv_out, index=False)
    positions.to_csv(pos_csv_out, index=False)
    #Save video
    #saveVideo(video_frames, arg.output)


#run using: python runTracking.py 0 (for live camera) data_output_path.csv --dist_profile dist_profile_path.csv (use "new" for new profile)
#note that video input should be adjusted since the way we take time is good only for real time, with video you will get much lower times depending on processing speed
#we did not have the time to fix this. you might fix it by using the frame number and the fps of the video to calculate the time (just for video input)
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="0 for live camera, 1 for video file and second arg is the path to the video file, third arg is the path to the output video file (second and third are optional)", type=int)
    parser.add_argument("--video_input", help="path to video file", type=str)
    parser.add_argument("output", help="path to output video file", type=str)
    parser.add_argument("--dist_profile", help="specify points for distortion profile", type=str)

    #if source is 1, then video_input is required
    if parser.parse_args().source == 1 and parser.parse_args().video_input == None:
        parser.error("source is 1, but video_input is not specified")

    if parser.parse_args().dist_profile == None:
        parser.parse_args().dist_profile = "configs/dist_profile.csv"

    args = parser.parse_args()
    main(args)