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
from keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout
from keras.models import Sequential, Model
from keras.utils import Sequence
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
parser.add_argument('-e', '--epochs', help='number of epochs')
parser.add_argument('-b, --batchsize', help='size of the batch')

# %%
try:
    args = parser.parse_args()
except:
    pass

# %%
labelPath = 'first_1k_labeled.csv'
# labelPath = 'full_labels.csv'
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
if args.path:
    vidPath = args.path
else:
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

    def __init__(self,
                 filenames,
                 labels,
                 batch_size,
                 max_frame_count,
                 frame_height=224,
                 frame_width=224,
                 n_channels=1):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.max_frame_count = max_frame_count
        self.h = frame_height
        self.w = frame_width
        self.n_channels = n_channels

    def __len__(self):
        return np.floor(len(self.filenames) / self.batch_size).astype(int)

    def __data_generation(self, idx_list):
        def padEmptyFrames(vid):
            num_frames = self.max_frame_count - vid.shape[0]
            empty_frames = np.empty(
                (num_frames, vid.shape[1], vid.shape[2]), dtype=np.float16)
            padded_vid = np.vstack((vid, empty_frames)).astype(np.float16)

            return padded_vid

        def getFrames(filepath, pad_frames=True):
            cap = cv2.VideoCapture(filepath)
            vid = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # can try going bigger up to 720*1080 later
                    gray = cv2.resize(gray, (self.h, self.w))
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

        x = np.empty((self.batch_size,
                      self.max_frame_count,
                      self.w,
                      self.h,
                      self.n_channels), dtype=np.float16)
        y = np.empty((self.batch_size), dtype=np.float16)
        for i, idx in enumerate(idx_list):
            file = self.filenames[idx]
            vid = getFrames(file)
            vid = vid.reshape(self.max_frame_count,
                              self.w,
                              self.h,
                              self.n_channels)
            x[i, ] = vid
            y[i, ] = self.labels[idx]
            # y = tf.keras.utils.to_categorical(y,
            #                                   num_classes=self.n_classes,
            #                                   dtype='float16')
        # print(x.shape, y.shape)
        return x, y

    def __getitem__(self, idx):
        batch = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        x, y = self.__data_generation(batch)
        return x, y


# %%
if args.batchsize:
    batch_size = args.batchsize
else:
    batch_size = 10
# TODO: re-examine with full dataset
class_weight = {
    0: 0.33,
    1: 0.67
}
frame_height = 64
frame_width = 64
# frame_height = 224
# frame_width = 224
n_channels = 1
# %%
x_train, x_test, y_train, y_test = train_test_split(
    filenames, labels, test_size=0.2)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
# %%
max_frame_count = getMaxFrameCount(filenames)
train_generator = DataGenerator(x_train,
                                y_train,
                                batch_size,
                                max_frame_count,
                                frame_height,
                                frame_width,
                                n_channels)
test_generator = DataGenerator(x_test,
                               y_test,
                               batch_size,
                               max_frame_count,
                               frame_height,
                               frame_width,
                               n_channels)
len(train_generator), len(test_generator)
# %%
input_shape = (None, frame_width, frame_height, n_channels)
if args.epochs:
    epochs = args.epochs
else:
    epochs = 1
class_weight = {
    0: 0.33,
    1: 0.67
}

model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(4, 4),
                     input_shape=input_shape,
                     batch_size=batch_size, data_format='channels_last',
                     padding='same', return_sequences=False,
                     dropout=0.2, recurrent_dropout=0.3))
model.add(BatchNormalization())
# model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
#                      padding='same', return_sequences=True,
#                      dropout=0.2, recurrent_dropout=0.3))
# model.add(BatchNormalization())
# model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
#                      padding='same', return_sequences=False,
#                      dropout=0.2, recurrent_dropout=0.3))
# model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta')
model.summary()

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
# Appendix


def padEmptyFrames(vid, max_frame_count):
    num_frames = max_frame_count - vid.shape[0]
    empty_frames = np.empty(
        (num_frames, vid.shape[1], vid.shape[2]), dtype='uint8')
    padded_vid = np.vstack((vid, empty_frames))

    return padded_vid


def getFrames(filepath, max_frame_count, pad_frames=True):
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
        if vid.shape[0] < max_frame_count:
            vid = padEmptyFrames(vid, max_frame_count)

    return vid


m = getMaxFrameCount(filenames)
batch_size = 10
idx = 0
# x = filenames[idx * batch_size:(idx + 1) * batch_size]
# y = labels[idx * batch_size:(idx + 1) * batch_size]
x = filenames
y = labels

x = np.array([getFrames(file, m) for file in tqdm(x)]) / 255
# x = x.reshape(self.batch_size, self.max_frame_count, 224, 224, 1)
b, f, w, h = x.shape
x = x.reshape(b, f, w, h, 1)
print(x.shape)
# %%
y = np.array(y)
x.shape, y.shape
hist = model.fit(x=x, y=y,
                 epochs=1,
                 verbose=1,
                 class_weight=class_weight)


# %%
