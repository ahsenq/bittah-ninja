# %% markdown
# Reading Video
simple video reader. reads in video frame by frame then stores those frames in a numpy array

# %%
import os
import numpy as np
import pandas as pd
import random
import cv2

# %%
datapath = '/tmp/vids'
vids = os.listdir(datapath)
# %%
# select a random video to test with
cap = cv2.VideoCapture(os.path.join(datapath, random.sample(vids, 1)[0]))
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        # FIXME: don't know why imshow doesn't work from jupyter lab...
        # cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
# %%
frames = np.array(frames)
frames.shape


# %%
