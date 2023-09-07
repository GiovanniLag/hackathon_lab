import cv2
import numpy as np
import requests
import imutils

def saveVideo(frames, path, fps=29):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(path, fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()

def startLiveCamera(camera_id):
    #open stream from camera
    cap = cv2.VideoCapture(camera_id)
    #block camera from changing auto exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
    #block camera from changing auto white balance
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.25)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 0.01)
    return cap

def redRemoteWebcam(url):
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    return img