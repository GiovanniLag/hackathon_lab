import cv2
from video_utils import startLiveCamera

def main():
    cap = cv2.startLiveCamera(0)
    counter = 0
    while True:
        if len(counter) == 5:
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        #take photo with spacebar
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite("test.jpg", frame)
            print("photo taken")
        cv2.imshow('frame', frame)
    cap.release()

if __name__ == "__main__":
    main()