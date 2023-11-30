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
	#file in which the data is saved. Some data lines are to be discarded due to malfunctionings of the camera
    data_file_path="trigger_chrono_data/times1.csv"


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
	
	#kalman filter
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.2
	
    def trigger(camName):
        global time_up
        global time_down
        color_treshold = 1
		#width of the stripe to be evalued later in order to detect the sphere:
		#w=5 for ball types 1 and 2 (the smaller ones)
		#w=10 for ball types 3 and 4 (the bigger ones)
        w=10
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
				
				#convert the frame to the gray color space
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (11, 11), 0)
				
				#get the difference between two frames. What is not moving will appear black, while objects that moved will appear white on camera
                frame_diff = cv2.absdiff(gray, prev_gray)
            
                _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
            
                kernel = np.ones((5,5),np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                #get array of the pixels in an orizontal stripe with an height of 2*w pixels and all the way wide
                line = thresh[int(frame_height/2)-w:int(frame_height/2)+w, :]
                #sum all the pixels in the line
                intensity_sum = np.sum(line)
                #normalize the sum between 0 and 255
                intensity_sum = (intensity_sum/255)/(2*w)
                #drow a circle in the frame to show intensity sum, more intense make it whiter
                color = int(intensity_sum*255)
                cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
				#if the sum of all the pixels increases above a certain threshold it means that the sphere has passed therethrough, so a circle is drawn 
				#in the middle of the camera window, the specific frame gets saved and the time variable is regstered
                if intensity_sum >= color_treshold:
                    print(f"color detected: {intensity_sum}")
                    cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
                    cv2.imshow("frame", thresh)
                    ddf = pd.read_csv('times1.csv')
                    index_num = ddf.shape[0]
                    cv2.imwrite("img/bottom_camera_" + str(index_num+1)  +".png",frame)
                    cap.release()
                    cv2.destroyWindow("frame")
					
                    time_down = time.time()
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
				
				
				#convert the frame to the gray color space
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (11, 11), 0)
				
				#get the difference between two frames. What is not moving will appear black, while objects that moved will appear white on camera
                frame_diff = cv2.absdiff(gray, prev_gray)
                frame_diff = cv2.flip(frame_diff,0)
                _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
            
                kernel = np.ones((5,5),np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                #get array of the pixels in an orizontal stripe with an height of 2*w pixels and all the way wide
                line = thresh[int(frame_height/2)-w:int(frame_height/2)+w, :]
                #sum all the pixels in the line
                intensity_sum = np.sum(line)
                #normalize the sum
                intensity_sum = (intensity_sum/255)/(2*w)
                #drow a circle in the frame to show intensity sum, more intense make it whiter
                color = int(intensity_sum*255)
                #show the frame
                cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
				#if the sum of all the pixels increases above a certain threshold it means that the sphere has passed therethrough, so a circle is drawn 
				#in the middle of the camera window, the specific frame gets saved and the time variable is regstered
                if intensity_sum >= color_treshold:
                    print(f"color detected: {intensity_sum}")
                    cv2.circle(thresh, (int(frame_width/2), int(frame_height/2)), 40, (color, color, color), 2)
                    cv2.imshow("frame2", thresh)
                    camera.close()
                    
                    frame = cv2.flip(frame,0)
                    ddf = pd.read_csv('times1.csv')
                    index_num = ddf.shape[0]
                    cv2.imwrite("img/top_camera_" + str(index_num+1) +".png",frame)
					
                    cv2.destroyWindow("frame2")
                    time_up = time.time()
                    return
                
                cv2.imshow("frame2", thresh)
                        
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
                time.sleep(0.1)
            
                prev_gray = gray

            
            cap.release()
            cv2.destroyAllWindows()
	
	#parallel execution of the trigger function, so that both of the cameras are open at the same time
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
    
