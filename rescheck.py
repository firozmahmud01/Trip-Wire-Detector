import cv2
import os
import cv2, numpy as np
d = np.load("lowrescalib.npz")

DISPLAY_WIDTH  = 1920        
DISPLAY_HEIGHT = 1080 

def resize_for_display(frame):
    h, w = frame.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h, 1.0)
    
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return frame

url="rtsp://admin:123456789!@192.168.0.150:554/ch1/sub"
# use this if gpu is missing
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

cap = cv2.VideoCapture(url)

retr,framm=cap.read()
retr,framm=cap.read()
h,w=framm.shape[:2]


newcameramtx, roi = cv2.getOptimalNewCameraMatrix(d['K'], d['dist'], (w, h), 1, (w, h))

while True:
    ret,frame=cap.read()
    if not ret:
        break

    dst = cv2.undistort(frame, d['K'], d['dist'])#, None, newcameramtx)

    # 6. Crop the image (using ROI from getOptimalNewCameraMatrix)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]


    cv2.imshow("image",resize_for_display(dst))

    cv2.waitKey(1)

# 2160, 3840 for "rtsp://admin:123456789!@192.168.0.150:554/StreamingChannels/101"
# 720, 1280 for "rtsp://admin:123456789!@192.168.0.150:554/Streaming/Channels/102"
# 720, 1280 for "rtsp://admin:123456789!@192.168.0.150:554/ch1/sub"
