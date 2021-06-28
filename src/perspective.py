import numpy as np
import cv2

fileName = 'alex_4640'
cap = cv2.VideoCapture('/home/alexandre/Desktop/temp/Videos/' + fileName + '.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
cap_width = cap.get(3)
cap_height = cap.get(4)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
out = cv2.VideoWriter('/home/alexandre/Desktop/temp/' + fileName + '.mp4', fourcc, fps, (960, 540))
frame_count = 0

print(total_frames)

while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    frame_count += 1

    pts1 = np.float32([[96, 72], [1160, 80], [84, 672], [1158, 692]])
    pts2 = np.float32([[0, 0], [960, 0], [0, 540], [960, 540]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (960, 540))

    out.write(result)

    cv2.imshow('frame',result)

    key = cv2.waitKey(1)
    #while key not in [ord('q'), ord('k'),ord('l')]:
    #    key = cv2.waitKey(0)
    if key == ord('l'):
        frame_count -= 5
        if frame_count < 0:
            frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_count)
    elif key == ord('q'):
        break

    
cap.release()
out.release()
cv2.destroyAllWindows()
