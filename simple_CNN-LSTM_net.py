# %%
import argparse
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Bidirectional, Conv2D, ConvLSTM2D, Dense,
                                     Dropout, Input, TimeDistributed)
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    os.chdir(os.path.join(os.getcwd(), 'FinalProject/bittah-ninja'))
    print(os.getcwd())
except:
    pass

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path to video set')

# %%
try:
    args = parser.parse_args()
except:
    pass

# %%
labelPath = 'full_labels.csv'
df = pd.read_csv(labelPath)
df['label'] = df['class']
df.drop(columns=['class'], inplace=True)
df.groupby('label').size()

# %%
df = df.loc[df.label != -1]
df.groupby('label').size()
# %%
df['punch'] = (df.label != 0).astype('int')
df.groupby('punch').size()


# %%
try:
    vidPath = args.path
except:
    vidPath = '../fullVidSet'

# %%
# def getDataFrame(labelPath):


def getMaxFrameCount(vidPath, df):
    frameCount = []
    for _, row in tqdm(df.iterrows()):
        vid = row.clip_title
        filepath = os.path.join(vidPath, vid)
        cap = cv2.VideoCapture(filepath)
        frameCount.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    return max(frameCount)
    # return frameCount


def padEmptyFrames(vid, maxFrameCount):
    numFrames = maxFrameCount - vid.shape[0]
    emptyFrames = np.zeros((numFrames, vid.shape[1], vid.shape[2]))
    paddedVid = np.vstack((vid, emptyFrames))

    return paddedVid


def getFrames(filepath, maxFrameCount):
    cap = cv2.VideoCapture(filepath)
    vid = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # can try going bigger up to 720*1080 later
            gray = cv2.resize(gray, (224, 224))
            vid.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    vid = np.array(vid)
    if vid.shape[0] < maxFrameCount:
        vid = padEmptyFrames(vid, maxFrameCount)

    return vid


def extractData(vidPath, df, maxFrameCount, startRow, endRow):
    data = []
    labels = []
    for _, row in tqdm(df.iloc[startRow:endRow].iterrows()):
        vid = row.clip_title
        labels.append(row.punch)
        filepath = os.path.join(vidPath, vid)
        vid = getFrames(filepath, maxFrameCount)
        data.append(vid)

    return data, labels


# %%
maxFrameCount = getMaxFrameCount(vidPath, df)
data, labels = extractData(vidPath, df, maxFrameCount, 0, 20)
len(data), len(labels)

# %%
import sys
sys.getsizeof(data)
# %%
sys.getsizeof(np.array(data))
# %%
try:
    with open('../allFrames.pickle', 'rb') as target:
        data, labels = pickle.load(target)
    if len(labels) != len(df.punch):
        maxFrameCount = getMaxFrameCount(vidPath, df)
        data, labels = extractData(vidPath, df, maxFrameCount)
        labels = np.array(labels)
        with open('../allFrames.pickle', 'wb') as f:
            pickle.dump((data, labels), f, protocol=pickle.HIGHEST_PROTOCOL)
except:
    maxFrameCount = getMaxFrameCount(vidPath, df)
    data, labels = extractData(vidPath, df, maxFrameCount)
    labels = np.array(labels)
    with open('../allFrames.pickle', 'wb') as f:
        pickle.dump((data, labels), f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
maxFrameCount = getMaxFrameCount(vidPath, df)
data, labels = extractData(vidPath, df, maxFrameCount)
labels = np.array(labels)
np.array(data).shape

# %%
import sys
sys.getsizeof(data)
# %%
m = getMaxFrameCount(vidPath, df)
sorted(m)
# %%
from collections import Counter
c = Counter(m)
c.most_common()


# %%
x_train, y_train, x_test, y_test = train_test_split(
    data, labels, test_size=0.2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
# model = Sequential()
# model.add(TimeDistributed(Conv2D(64, (3, 3)),
#                           input_shape=(10, 299, 299, 3)))
# model.summary()
# inputs = Input(input_shape=(224, 224))
class_weight = {
    0: 0.33,
    1: 0.67
}
model = Sequential()
model.add(Bidirectional(ConvLSTM2D(filters=64,
                                   kernel_size=(3, 3),
                                   input_shape=(None, 224, 224, 1),
                                   padding='same',
                                   dropout=0.1,
                                   recurrent_dropout=0.1,
                                   activation='relu')))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              mertics=['acc'])
# model.summary()

# %%
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=1,
          batch_size=16,
          class_weight=class_weight)

# %%
# seq = Sequential()
# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    input_shape=(None, 40, 40, 1),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))
# seq.compile(loss='binary_crossentropy', optimizer='adadelta')

# %%
# Appendix
# img = allFrames[0][0]
# img.shape
# #%%
# cv2.imshow('frame',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #%%
# h,w = 720,1080
# wScale = w/singleFrame.shape[1]
# hScale = w/singleFrame.shape[0]
# #%%
# img = cv2.resize(singleFrame, (1080, 720))
# img.shape
