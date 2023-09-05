#Kalman filter based tracker

class kTracker():
    def __init__(frame, window, camera=0) -> None:
        self.frame = frame
        self.window = window
        self.cap = cv2.VideoCapture(camera)

        # Initialize the histogram.
        x, y, w, h = track_window
        roi = hsv_frame[y:y+h, x:x+w]
        roi_hist = cv2.calcHist([roi], [0, 2], None, [15, 16],[0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        #initialize kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.03
        
        
        cx = np.array([np.float32(x+w/2), np.float32(y+h/2)])
        cy = np.array([np.float32(x+w/2), np.float32(y+h/2)])

        self.kalman.statePre = np.array([cx, cy, 0, 0], np.float32)
        self.kalman.statePost = np.array([cx, cy, 0, 0], np.float32)
        
        pass