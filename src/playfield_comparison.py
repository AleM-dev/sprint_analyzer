import numpy as np
import cv2

fileName = 'reddy38'
cap = cv2.VideoCapture('/home/alexandre/Desktop/temp/Videos/' + fileName + '.mp4')
    
cap_width = int(cap.get(3))
cap_height = int(cap.get(4))
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc(*'mpeg')

fps = 30
frame_width = 1600
frame_height = 900
   
out = cv2.VideoWriter('/home/alexandre/Desktop/temp/' + fileName + '.avi', fourcc, fps, (frame_width, frame_height))

start_frame = 92
duration = 1098
frame_i = 83
count = 0

frame_offsets = [0, 1117, 1081, 1092, 1086]

while cap.isOpened() and count < 150:
    print(frame_i)
    out_frame = np.zeros((frame_height,frame_width,3), np.uint8)
    frame_temp = frame_i

    cv2.rectangle(out_frame, (0, frame_height-108-66), (frame_width, frame_height-66), (48,0,0), -1) 
    cv2.rectangle(out_frame, (0, frame_height-66), (frame_width, frame_height), (0,0,32), -1) 

    if frame_i >= start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, frame = cap.read()
        out_frame[260+210: 260+210+54, 260-116-72: 260-72] = frame[0: 54, cap_width-116: cap_width]
        out_frame[frame_height-108-66: frame_height-66, 0: 240] = frame[0: 108, 480: 720]
        out_frame[frame_height-66: frame_height, 0: 240] = frame[474-42: 540-42, 480: 720]        

    if frame_i > duration:
        frame_i = duration
        count += 1

    for i in range(len(frame_offsets)):
        temp = frame_offsets[i]
        frame_temp = frame_temp + temp

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_temp)
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        board = frame[62: 471, 78: 414]
        resized_board = cv2.resize(board, (216, 262)) 

#        cv2.imshow('frame', resized_board)
    

        offset_x = (i)*216+260
        

        if frame_i >= start_frame:
            out_frame[frame_height-cap_height -108-66: frame_height-108-66, offset_x + 50: offset_x + 166] = frame[0: frame_height, cap_width-116: cap_width]
            out_frame[frame_height-108-66: frame_height-66, offset_x + 50: offset_x + 166] = frame[0: 108, 726: 842]
            out_frame[frame_height-66: frame_height, offset_x: offset_x + 166] = frame[474-42: 540-42, 676: 842]
        out_frame[0: 262, offset_x: offset_x + 216] = resized_board

    
    cv2.imshow('frame', out_frame)
    key = cv2.waitKey(1)
    #while key not in [ord('q'), ord('k'),ord('l')]:
    #    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    out.write(out_frame)
    frame_i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
    
