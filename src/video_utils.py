import numpy as np
import cv2

fileName = 'col_o'
cap = cv2.VideoCapture('/home/alexandre/Desktop/temp/Videos/' + fileName + '.mp4')
    
cap_width = int(cap.get(3))
cap_height = int(cap.get(4))
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc(*'mpeg')

fps = 30
   
count = 0
out = cv2.VideoWriter('/home/alexandre/Desktop/temp/' + fileName + '.avi', fourcc, fps, (cap_width, cap_height))
frame_count = 0

while cap.isOpened() and count < 3*fps:
    if frame_count < total_frames - 1:
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    else:
        count += 1

    key = cv2.waitKey(1)
    #while key not in [ord('q'), ord('k'),ord('l')]:
    #    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    out.write(frame)
#    cv2.imshow('frame',frame)
        

cap.release()
out.release()
cv2.destroyAllWindows()
    