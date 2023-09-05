import cv2
import numpy as np
import matplotlib.pyplot as plt

def frame_diff(prev_frame, cap):
    ret, next_frame = cap.read()
    diff = cv2.absdiff(prev_frame, next_frame)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return diff

def frameCenter(frame, area):
    # Find contours in the bw frame
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > area: # Minimum contour area threshold
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
                print(f"Centroid of dot: (x, y) = ({center_x:.1f}, {center_y:.1f})")
                return (center_x, center_y)
    return (-1, -1)


def ballIsMiddle(center_y, frame_height, treshold=3):
    if center_y < frame_height/2 + treshold and center_y > frame_height/2 - treshold:
        return True
    else:
        return False




def main():
    #open stream from camera
    cap = cv2.VideoCapture(1)
    #block camera from changing auto exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
    #block camera from changing auto white balance
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.25)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 0.01)

    #create window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    
    #get frame sizes
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #when q is pressed, quit
    while True:
        ret, frame = cap.read()
        #compute the absolute difference between the current frame and the next frame
        result = frame_diff(frame, cap)
        #find centroid of ball
        center_x,center_y = frameCenter(result, 4000)
        #make result rgb so we can add a green circle
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        #add centroid to frame as a green circle
        cv2.circle(result, (int(center_x), int(center_y)), 10, (0, 255, 0), 1)

        #if ball is in the middle draw a line across the orizontal center of the frame
        if ballIsMiddle(center_y, frame_height):
            cv2.line(result, (0, 240), (640, 240), (255, 0, 0), 2)


        #show the result
        cv2.imshow('frame', result)
        if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
            cap.release()
            cv2.destroyAllWindows()
            break

    return


if __name__ == '__main__':
    main()