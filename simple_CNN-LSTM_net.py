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
                                     Dropout, Input, TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
filenames = [os.path.join(vidPath, f) for f in df.clip_title]
labels = df.punch.tolist()

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
        return np.ceil(len(self.filenames) / float(self.batch_size))

    def __getitem__(self, idx):

        def padEmptyFrames(vid):
            num_frames = self.max_frame_count - vid.shape[0]
            empty_frames = np.empty(
                (num_frames, vid.shape[1], vid.shape[2]), dtype='uint8')
            padded_vid = np.vstack((vid, empty_frames))

            return padded_vid

        def getFrames(filepath, pad_frames=True):
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
            if pad_frames:
                if vid.shape[0] < self.max_frame_count:
                    vid = padEmptyFrames(vid)

            return vid

        x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([getFrames(file) for file in x]), np.array(y)

    # def extractData(filenames, max_frame_count, pad_frames=True):
    #     data = []
    #     labels = []
    #     for _, row in tqdm(df.iterrows()):
    #         vid = row.clip_title
    #         labels.append(row.punch)
    #         filepath = os.path.join(vidPath, vid)
    #         vid = getFrames(filepath, max_frame_count, pad_frames=pad_frames)
    #         data.append(vid)

    #     return data, labels
# %%
# def getDataFrame(labelPath):

    # return frameCount


# %%
# Appendix
# vid = df.clip_title[0]
# filepath = os.path.join(vidPath, vid)
# # %%
# frames1 = getFrames(filepath, 229, pad_frames=False)
# print(np.array(frames1).shape, sys.getsizeof(np.array(frames1)))
# # %%
# frames2 = getFrames(filepath, 229, pad_frames=True)
# print(np.array(frames2).shape, sys.getsizeof(np.array(frames2)))
# # %%
# len(frames1) / len(frames2), sys.getsizeof(np.array(frames1)) / \
#     sys.getsizeof(np.array(frames2))
# # %%
# testFrame = np.vstack((np.array(frames1), np.array(frames1)))
# testFrame.shape, sys.getsizeof(testFrame)
# # %%
# z1 = np.zeros_like(np.array(frames1))
# sys.getsizeof(z1)
# # %%
# sys.getsizeof(np.vstack((np.array(frames1), z1)))
# # %%
# z1.dtype
# %%
# max_frame_count = getMaxFrameCount(vidPath, df)
# data, labels = extractData(vidPath, df, max_frame_count, pad_frames=True)
# sys.getsizeof(data)  # sys.getsizeof(np.array(data))
# # %%
# max_frame_count = getMaxFrameCount(vidPath, df)
# slicer = (0, 10)
# data, labels = extractData(vidPath, df, max_frame_count,
#                            slicer=slicer, pad_frames=True)
# sys.getsizeof(data), sys.getsizeof(np.array(data))
# %%
# try:
#     with open('../allFrames.pickle', 'rb') as target:
#         data, labels = pickle.load(target)
#     if len(labels) != len(df.punch):
#         max_frame_count = getMaxFrameCount(vidPath, df)
#         data, labels = extractData(vidPath, df, max_frame_count)
#         data = np.array(data)
#         labels = np.array(labels)
#         with open('../allFrames.pickle', 'wb') as f:
#             pickle.dump((data, labels), f, protocol=pickle.HIGHEST_PROTOCOL)
# except:
#     max_frame_count = getMaxFrameCount(vidPath, df)
#     data, labels = extractData(vidPath, df, max_frame_count)
#     data = np.array(data)
#     labels = np.array(labels)
#     with open('../allFrames.pickle', 'wb') as f:
#         pickle.dump((data, labels), f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# max_frame_count = getMaxFrameCount(vidPath, df)
# data, labels = extractData(vidPath, df, max_frame_count)
# labels = np.array(labels)
# data = np.array(data)

# sys.getsizeof(data)

# c = Counter(m)
# c.most_common()


# %%
x_train, x_test, y_train, y_test = train_test_split(
    filenames, labels, test_size=0.2)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# %%
m = getMaxFrameCount(filenames)
train_generator = DataGenerator(x_train, y_train, 10, m)
test_generator = DataGenerator(x_test, y_test, 10, m)

# %%
batch_size = 10


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
hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=(len(x_train) // batch_size),
                           epochs=1,
                           verbose=1,
                           validation_data=test_generator,
                           validation_steps=(len(x_test) // batch_size),
                           class_weight=class_weight,
                           use_multiprocessing=True)

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
