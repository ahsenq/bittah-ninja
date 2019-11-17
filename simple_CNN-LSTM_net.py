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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (BatchNormalization, ConvLSTM2D, Dense,
                                     Dropout, Flatten)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

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
                elif vid.shape[0] >= self.max_frame_count:
                    vid = vid[:self.max_frame_count]

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
        y = tf.keras.utils.to_categorical(
            y, num_classes=len(set(self.labels)), dtype='float16')
        return x, y

    def __getitem__(self, idx):
        batch = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        x, y = self.__data_generation(batch)
        return x, y


# %%
if __name__ == "__main__":
    assert tf.__version__.startswith(
        '2'), 'you need to upgrade to tensorflow 2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--vidpath', default='vids/scaled')
    parser.add_argument('--modelpath', default='/data/models')
    parser.add_argument('--labelpath', default='week10_labeled.csv')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--use_cache', action='store_true', default=False)
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(
            ['--vidpath=/data/vids/scaled', '--use_cache'])
    print(args)

    labelPath = args.labelpath
    df = pd.read_csv(labelPath)
    df.head()

    new_files = []
    for file in df.clip_title:
        newfile = ''.join(file.split('.mp4')) + '.mp4'
        new_files.append(newfile)
    df['clip_title'] = new_files
    df['label'] = df['class']
    df.drop(columns=['class'], inplace=True)
    df.groupby('label').size()

    df = df.loc[df.label != -1]
    df = df.loc[df.label != 99]
    df.groupby('label').size()

    vidPath = args.vidpath
    filenames = [f.split('.mp4')[0] + '_scaled.mp4' for f in df.clip_title]
    filenames = [os.path.join(vidPath, f) for f in filenames]
    labels = df.label.tolist()
    batch_size = args.batch_size
    class_weight = compute_class_weight('balanced', np.unique(labels), labels)

    if args.use_cache:
        print('using cached train and test sets')
        with open(os.path.join(args.modelpath, f'train.pickle'), 'rb') as f:
            x_train, y_train = pickle.load(f)
        with open(os.path.join(args.modelpath, f'test.pickle'), 'rb') as f:
            x_test, y_test = pickle.load(f)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            filenames, labels, test_size=0.2)

        modelpath = args.modelpath
        os.makedirs(modelpath, exist_ok=True)
        with open(os.path.join(modelpath, f'train.pickle'), 'wb') as f:
            pickle.dump((x_train, y_train), f, protocol=-1)
        with open(os.path.join(modelpath, f'test.pickle'), 'wb') as f:
            pickle.dump((x_test, y_test), f, protocol=-1)

    print(len(x_train), len(y_train))
    print(len(x_test), len(y_test))

    # max_frame_count = getMaxFrameCount(filenames)
    max_frame_count = 151
    print('max number of frames: ', max_frame_count)
    train_generator = DataGenerator(x_train,
                                    y_train,
                                    batch_size,
                                    max_frame_count)
    test_generator = DataGenerator(x_test,
                                   y_test,
                                   batch_size,
                                   max_frame_count)
    len(train_generator), len(test_generator)

    input_shape = (None,
                   train_generator.w,
                   train_generator.h,
                   train_generator.n_channels)
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    model.add(ConvLSTM2D(filters=8, kernel_size=(4, 4),
                         input_shape=input_shape, data_format='channels_last',
                         padding='same', return_sequences=True,
                         dropout=0.2, recurrent_dropout=0))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.2, recurrent_dropout=0))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=8, kernel_size=(2, 2),
                         padding='same', return_sequences=False,
                         dropout=0.2, recurrent_dropout=0))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(set(labels)), activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta')
    model.summary()

    cp_dir = os.path.join(args.modelpath, 'checkpoints')
    os.makedirs(cp_dir, exist_ok=True)
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cp_dir, 'mymodel_{epoch}.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_freq='epoch'
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    hist = model.fit_generator(generator=train_generator,
                               steps_per_epoch=(
                                   len(x_train) // batch_size),
                               epochs=args.epochs,
                               verbose=1,
                               validation_data=test_generator,
                               validation_steps=(
                                   len(x_test) // batch_size),
                               class_weight=class_weight,
                               use_multiprocessing=True,
                               callbacks=[es, cp])
