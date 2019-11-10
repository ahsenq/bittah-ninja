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
from tqdm import tqdm

import keras
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Model, Sequential
from keras.utils import Sequence

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--vidpath', default='vids/scaled')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=32, type=int)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

# %%
# labelPath = 'bittah-ninja/first_1k_labeled_long_vids_removed.csv'
labelPath = 'first_1k_labeled_long_vids_removed.csv'
# labelPath = 'full_labels.csv'
df = pd.read_csv(labelPath)
new_files = []
for file in df.clip_title:
    newfile = ''.join(file.split('.mp4')) + '.mp4'
    new_files.append(newfile)
df['clip_title'] = new_files
# df['label'] = df['class']
# df.drop(columns=['class'], inplace=True)
df.groupby('label').size()

# %%
df = df.loc[df.label != -1]
df.groupby('label').size()
# %%
df.shape
# %%
# df['punch'] = (df.label != 0).astype('int')
# df.groupby('punch').size()

# %%
vidPath = args.vidpath
filenames = [f.split('.mp4')[0] + '_scaled.mp4' for f in df.clip_title]
filenames = [os.path.join(vidPath, f) for f in filenames]
# labels = df.punch.tolist()
labels = df.label.tolist()

# %%
# def getMaxFrameCount(filenames):
#     frameCount = []
#     for file in tqdm(filenames):
#         cap = cv2.VideoCapture(file)
#         frameCount.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

#     return max(frameCount)


class DataGenerator(Sequence):

    def __init__(self,
                 filenames,
                 labels,
                 batch_size,
                 frame_height=224,
                 frame_width=224,
                 n_channels=1):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.h = frame_height
        self.w = frame_width
        self.n_channels = n_channels

    def __len__(self):
        return np.ceil(len(self.filenames) / self.batch_size).astype(int)

    def __data_generation(self, idx_list):
        def getSingleFrame(filepath):
            cap = cv2.VideoCapture(filepath)
            vid = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (self.h, self.w))
                    vid.append(gray)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            j = int(np.random.choice(len(vid), 1))
            frame = vid[j]
            return frame

        x = np.empty((self.batch_size,
                      self.w,
                      self.h,
                      self.n_channels), dtype=np.float16)
        y = np.empty((self.batch_size), dtype=np.float16)
        for i, idx in enumerate(idx_list):
            file = self.filenames[idx]
            frame = getSingleFrame(file)
            frame = frame.reshape(self.w,
                                  self.h,
                                  self.n_channels)
            x[i, ] = frame
            y[i, ] = self.labels[idx]
            y = tf.keras.utils.to_categorical(y,
                                              num_classes=len(set(self.labels)),
                                              dtype='float16')
        # print(x.shape, y.shape)
        return x, y
        # yield x, y

    def __getitem__(self, idx):
        batch = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        x, y = self.__data_generation(batch)
        return x, y
        # yield x, y


# %%
batch_size = args.batch_size
labels_counts = Counter(labels)
# TODO: Not sure if this should be 1 - x or x
class_weight = {k:1-(v/len(labels)) for k,v in labels_counts.items()}
# %%
x_train, x_test, y_train, y_test = train_test_split(
    filenames, labels, test_size=0.2)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
# %%
# max_frame_count = getMaxFrameCount(filenames)
train_generator = DataGenerator(x_train,
                                y_train,
                                batch_size)
test_generator = DataGenerator(x_test,
                               y_test,
                               batch_size)
len(train_generator), len(test_generator)
# %%
# lab, batch = next(train_generator)
# %%

# %%

# %%
input_shape = (224, 224, 1)
epochs = args.epochs
# model = Sequential()
inputs = keras.layers.Input(shape=input_shape, name='inputs')
conv = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
conv = Conv2D(32, (3, 3), activation='relu', padding='same')(pool)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
# dropout = Dropout(0.25)(pool)
conv = Conv2D(32, (4, 4), activation='relu', padding='same')(pool)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
conv = Conv2D(32, (4, 4), activation='relu', padding='same')(pool)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
dropout = Dropout(0.25)(pool)
flat = Flatten()(dropout)
dense = Dense(16, activation='relu')(flat)
dropout = Dropout(0.25)(dense)
outputs = Dense(len(set(labels)), activation='softmax', name='outputs')(dropout)
model = Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# %%
# inputs = keras.layers.Input(shape=input_shape, name='inputs')
# conv2D = keras.layers.Conv2D(1, kernel_size=(3,3))
# conv3 = Conv3D(4, (3, 3, 3), strides=(1, 1, 1), padding='same',
#                data_format='channels_last', activation='relu')(inputs)
# pool = keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
#                                  strides=(2, 2, 2), padding='same',
#                                  data_format='channels_last')(conv3)
# drop = Dropout(0.5)(pool)
# flat = Flatten()(drop)
# # model.add(Dense(128, activation='relu'))
# # model.add(Dense(64, activation='relu'))
# outputs = Dense(1, activation='sigmoid')(flat)
# model = Model(inputs, outputs)


# %%

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
# Appendix
# %%


def getSingleFrame(filepath):
    cap = cv2.VideoCapture(filepath)
    vid = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (h, w))
            vid.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    frame = np.random.choice(np.array(vid), 1)
    return frame


# %%
batch_size = 4
w = 224
h = 224
n_channels = 1
idx_list = range(10)
x = np.empty((batch_size,
              w,
              h,
              n_channels), dtype=np.float16)
y = np.empty((batch_size), dtype=np.float16)
for i, idx in enumerate(idx_list):
    file = filenames[idx]
    cap = cv2.VideoCapture(file)
    vid = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (h, w))
            vid.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    j = int(np.random.choice(len(vid), 1))
    frame = vid[j]
    # frame = getSingleFrame(file)
    frame = frame.reshape(w,
                          h,
                          n_channels)
    x[i, ] = frame
    y[i, ] = labels[idx]

# %%
filenames2 = [os.path.join(vidPath, f) for f in os.listdir(vidPath)]
[f for f in filenames if f in filenames2]

# %%
!pwd
