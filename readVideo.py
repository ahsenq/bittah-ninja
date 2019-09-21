# %% markdown
# Reading Video
# %%
import os
import numpy as np
import pandas as pd
import cv2 as cv

# %%
datapath = '/tmp/vids'
vids = os.listdir(datapath)
cap = cv.VideoCapture(os.path.join(datapath, vids[0]))
if not cap.isOpened():
    print('did not successfully load the video')

# %%
_, frame = cap.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
cv.imshow('vid', gray)
