import cv2

def saveVideo(frames, path, fps=29):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(path, fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()