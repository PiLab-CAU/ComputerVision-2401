import cv2
import numpy as np
from utils import lucas_kanade_optical_flow, lucas_kanade_optical_flow_pyr
from utils import draw_optical_flow_arrows

import time


# capturing video - dataset from TinyVirat dataset: 
# https://www.crcv.ucf.edu/research/projects/tinyvirat-low-resolution-video-action-recognition/
cap = cv2.VideoCapture('6020.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Frame size: {frame_width}x{frame_height}")
print(f"FPS: {fps}")

#save the results in video form
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('6020_lkpr.mp4', fourcc, fps, (frame_width, frame_height))


#read first frame
ret, frame = cap.read()
I1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

frame_num = 0
while True:
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    I2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    #u, v = lucas_kanade_optical_flow(I1, I2)
    u, v = lucas_kanade_optical_flow_pyr(I1, I2)
    toc = time.time()
    print(f'fetching frame {frame_num}, elapsed time {toc-tic:.4f}')


    vis_lk = draw_optical_flow_arrows(I1, u, v)
    out.write(vis_lk.astype(np.uint8))
    frame_num+=1
    I1 = I2

out.release()


