# %%
import argparse
import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Bidirectional, Conv2D, ConvLSTM2D, Dense,
                                     Dropout, Input, TimeDistributed, MaxPooling2D, LSTM, Flatten)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    os.chdir(os.path.join(os.getcwd(), 'FinalProject/bittah-ninja'))
    print(os.getcwd())
except:
    pass

# %%
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--path', help='path to video set')

# %%
# try:
#     args = parser.parse_args()
# except:
#     pass

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
# try:
#     vidPath = args.path
# except:
#     vidPath = '../fullVidSet'
vidPath = '../fullVidSet'
filenames = [os.path.join(vidPath, f) for f in df.clip_title]
labels = df.punch.tolist()

# %%
testFiles = [f for f in filenames if 'V_' in f]
testFiles[0]

# %%
filepath = testFiles[0]
filepath
# %%


def padEmptyFrames(vid, max_frame_count=229):
    num_frames = max_frame_count - vid.shape[0]
    empty_frames = np.empty(
        (num_frames, vid.shape[1], vid.shape[2]), dtype='uint8')
    padded_vid = np.vstack((vid, empty_frames))

    return padded_vid


# %%
# playground
pad_frames = True
max_frame_count = 229
cap = cv2.VideoCapture(filepath)
vid = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # can try going bigger up to 720*1080 later
        # gray = cv2.resize(gray, (224, 224))
        gray = cv2.resize(gray, (20, 20))
        vid.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
vid = np.array(vid)
if pad_frames:
    if vid.shape[0] < max_frame_count:
        vid = padEmptyFrames(vid)
vid.shape
# %%

# %%


def getMaxFrameCount(filenames):
    frameCount = []
    for file in tqdm(filenames):
        cap = cv2.VideoCapture(file)
        frameCount.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    return max(frameCount)


class DataGenerator(Sequence):

    def __init__(self, filenames, labels, batch_size, max_frame_count):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.max_frame_count = max_frame_count

    def __len__(self):
        return np.ceil(len(self.filenames) / self.batch_size).astype(int)

    def __getitem__(self, idx):

        def padEmptyFrames(vid):
            num_frames = self.max_frame_count - vid.shape[0]
            empty_frames = np.empty(
                (num_frames, vid.shape[1], vid.shape[2]), dtype='uint8')
            padded_vid = np.vstack((vid, empty_frames))

            return padded_vid

        def getFrames(filepath, pad_frames=True):
            # print(filepath)
            cap = cv2.VideoCapture(filepath)
            vid = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # can try going bigger up to 720*1080 later
                    # gray = cv2.resize(gray, (224, 224))
                    gray = cv2.resize(gray, (20, 20))
                    vid.append(gray)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            vid = np.array(vid)
            if pad_frames:
                if vid.shape[0] < self.max_frame_count:
                    vid = padEmptyFrames(vid)

            return vid

        x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.array([getFrames(file) for file in x]) / 255
        # x = x.reshape(self.batch_size, self.max_frame_count, 224, 224, 1)
        x = x.reshape(self.batch_size, self.max_frame_count, 20, 20, 1)

        return x, np.array(y)


# %%
x_train, x_test, y_train, y_test = train_test_split(
    filenames, labels, test_size=0.2)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# %%
m = getMaxFrameCount(filenames)
train_generator = DataGenerator(x_train, y_train, 10, m)
test_generator = DataGenerator(x_test, y_test, 10, m)
len(train_generator), len(test_generator)
# %%
batch_size = 2
class_weight = {
    0: 0.33,
    1: 0.67
}
model = Sequential()
# CNN
model.add(TimeDistributed(Conv2D(filters=1,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 #  input_shape=(None, 224, 224, 1),
                                 input_shape=(20, 20, 1),
                                 padding='same',
                                 activation='relu',
                                 data_format='channels_last')))
model.add(TimeDistributed(MaxPooling2D(strides=(1, 1))))
model.add(TimeDistributed(Flatten()))
# LSTM
model.add(LSTM(32, activation='relu', recurrent_activation='linear'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Bidirectional(ConvLSTM2D(filters=2,
#                                    kernel_size=(3, 3),
#                                    input_shape=(None, 224, 224, 1),
#                                    padding='same',
#                                    dropout=0.1,
#                                    recurrent_dropout=0.1,
#                                    activation='relu')))

# model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              mertics=['acc'])
# model.summary()

# %%
hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=(len(x_train) // batch_size),
                           epochs=1,
                           verbose=1,
                           validation_data=test_generator,
                           validation_steps=(len(x_test) // batch_size),
                           class_weight=class_weight,
                           use_multiprocessing=False)


# %%
