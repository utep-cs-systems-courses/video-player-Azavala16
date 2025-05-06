import cv2

cap = cv2.VideoCapture('clip.mp4')
if cap.isOpened():
    print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
else:
    print("Failed to open video.")
cap.release()

