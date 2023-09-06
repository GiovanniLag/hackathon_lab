import cv2
import picamera2
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import matplotlib.pyplot as plt

time_up = None
time_down = None

def main(args):

    ball_type = args.ball_type
    
    dataframe = pd.DataFrame(columns=["ball_type","delta_t"], index=None)
    data_file_path="times.csv"


    def saveVideo(frames, path, fps=29):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (frames[0].shape[1], frames[0].shape[0])
        out = cv2.VideoWriter(path, fourcc, fps, size)
        for frame in frames:
            out.write(frame)
        out.release()

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
    def trigger(camName):
        global time_up
        global time_down
        color_treshold = 2
        w=5
        if(camName=="std"):
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cap = cv2.VideoCapture(2)
            ret, prev_frame = cap.read()

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)
            
            #init kalman filter, first frame
            init = False
            
            #to save video
            video_frames = []
            
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"frmae height is{frame_height}")
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
                line = thresh[int(frame_height/2)-w:int(frame_height/2)+w, :]
                #sum all the pixels in the line
                intensity_sum = np.sum(line)
                #normalize the sum between 0 and 255
                intensity_sum = (intensity_sum/255)/(2*w)
                #drow a circle in the frame to show intensity sum, more intense make it whiter
                color = int(intensity_sum*255)
                cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
                if intensity_sum >= color_treshold:
                    print(f"color detected: {intensity_sum}")
                    cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
                    cv2.imshow("frame", thresh)
                    cv2.imwrite("test_up.png",frame)
                    cap.release()
                    cv2.destroyWindow("frame")
                    #print(f"START AT:{time.time()}")
                    time_up = time.time()
                    return
                #show the frame
                cv2.imshow("frame", thresh)
                
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
                time.sleep(0.1)
            
                prev_gray = gray
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
            camera = picamera2.Picamera2()
            camera.start()
            prev_frame = camera.capture_array('main')
            frame_width = 640
            frame_height = 480
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)
            
            #init kalman filter, first frame
            init = False
            
            #to save video
            video_frames = []
            print(f"frmae height is{frame_height}")
            # while cap.isOpened():
            while(True):
                intensity_sum = 0
                frame = camera.capture_array('main')
                # if not ret:
                #     break
            
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (11, 11), 0)
            
                frame_diff = cv2.absdiff(gray, prev_gray)
                frame_diff = cv2.flip(frame_diff,0)
                _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
            
                kernel = np.ones((5,5),np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                
                line = thresh[int(frame_height/2)-w:int(frame_height/2)+w, :]
                #sum all the pixels in the line
                intensity_sum = np.sum(line)
                #normalize the sum between 0 and 255
                intensity_sum = (intensity_sum/255)/(2*w)
                #drow a circle in the frame to show intensity sum, more intense make it whiter
                color = int(intensity_sum*255)
                #show the frame
                cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
                if intensity_sum >= color_treshold:
                    print(f"color detected: {intensity_sum}")
                    cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
                    cv2.imshow("frame2", thresh)
                    camera.close()
                    
                    frame = cv2.flip(frame,0)
                    cv2.imwrite("test_down.png",frame)
                    #print(f"START AT:{time.time()}")
                    cv2.destroyWindow("frame2")
                    time_down = time.time()
                    return
                
                cv2.imshow("frame2", thresh)
                        
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
                time.sleep(0.1)
            
                prev_gray = gray

            
            cap.release()
            cv2.destroyAllWindows()

    params_list = ["std", ""]
    with ThreadPoolExecutor() as executor:
        executor.map(trigger, params_list)
        
    cv2.destroyAllWindows()

    dataframe = pd.concat([result, pd.DataFrame({'ball_type': ball_type, 'delta_t': time_down-time_up}, index=[0])], ignore_index=True)

    if not os.path.exists(data_file_path):
        dataframe.to_csv(data_file_path, index=False)
    else:
        dataframe.to_csv(data_file_path, mode='a', header=False, index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ball_type")
    
    args = parser.parse_args()
    main(args)
    
