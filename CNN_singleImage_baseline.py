# %%
# Only execute if running in container
# !apt install libsm6 libxext6 libxrender-dev git -y
# !pip3 install pandas scikit-learn opencv-python tqdm


# %%
import argparse
import os
import pickle
import sys
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
# %%


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
        return np.floor(len(self.filenames) / self.batch_size).astype(int)

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
        y = np.empty(self.batch_size, dtype=np.float16)
        for i, idx in enumerate(idx_list):
            file = self.filenames[idx]
            frame = getSingleFrame(file)
            frame = frame.reshape(self.w,
                                  self.h,
                                  self.n_channels)
            x[i, ] = frame
            y[i, ] = self.labels[idx]
        y = tf.keras.utils.to_categorical(
            y, num_classes=len(set(self.labels)), dtype='float16')
        return x, y

    def __getitem__(self, idx):
        batch = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        x, y = self.__data_generation(batch)
        return x, y

if __name__ == "__main__":
    # %%
    print(tf.__version__)
    print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    # %%
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
        # may need to modify vidpath depending on where you are running the script
        #     args = parser.parse_args(['--vidpath=/tf/data/vids/scaled', '--batch_size=4'])
        args = parser.parse_args(['--vidpath=/data/vids/scaled', '--use_cache'])
    print(args)

    # %%
    # labelPath = 'bittah-ninja/first_1k_labeled_long_vids_removed.csv'
    labelPath = args.labelpath
    # labelPath = 'full_labels.csv'
    df = pd.read_csv(labelPath)
    df.head()
    # %%
    new_files = []
    for file in df.clip_title:
        newfile = ''.join(file.split('.mp4')) + '.mp4'
        new_files.append(newfile)
    df['clip_title'] = new_files
    df['label'] = df['class']
    df.drop(columns=['class'], inplace=True)
    df.groupby('label').size()
    # %%
    df = df.loc[df.label != -1]
    df = df.loc[df.label != 99]
    df.groupby('label').size()

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
    batch_size = args.batch_size
    class_weight = compute_class_weight('balanced', np.unique(labels), labels)

    # %%
    if args.use_cache:
        print('using cached train and test sets')
        with open(os.path.join(args.modelpath, f'train.pickle'), 'rb') as f:
            x_train, y_train = pickle.load(f)
        with open(os.path.join(args.modelpath, f'test.pickle'), 'rb') as f:
            x_test, y_test = pickle.load(f)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            filenames, labels, test_size=0.2)
        # print(len(x_train), len(y_train))
        # print(len(x_test), len(y_test))
        modelpath = args.modelpath
        os.makedirs(modelpath, exist_ok=True)
        with open(os.path.join(modelpath, f'train.pickle'), 'wb') as f:
            pickle.dump((x_train, y_train), f, protocol=-1)
        with open(os.path.join(modelpath, f'test.pickle'), 'wb') as f:
            pickle.dump((x_test, y_test), f, protocol=-1)

    # %%
    train_generator = DataGenerator(x_train, y_train, batch_size)
    test_generator = DataGenerator(x_test, y_test, batch_size)
    len(train_generator), len(test_generator)

    # %%
    input_shape = (224, 224, 1)
    epochs = args.epochs
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
    dropout = Dropout(0.5)(pool)
    flat = Flatten()(dropout)
    dense = Dense(16, activation='relu')(flat)
    dropout = Dropout(0.25)(dense)
    outputs = Dense(len(set(labels)), activation='softmax',
                    name='outputs')(dropout)
    model = Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()


    # %%
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    hist = model.fit_generator(generator=train_generator,
                               steps_per_epoch=(len(x_train) // batch_size),
                               epochs=args.epochs,
                               verbose=1,
                               validation_data=test_generator,
                               validation_steps=(len(x_test) // batch_size),
                               shuffle=False,
                               class_weight=class_weight,
                               use_multiprocessing=False,
                               callbacks=[es])


    # %%
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    # modelpath = args.modelpath
    # os.makedirs(modelpath, exist_ok=True)
    model.save(os.path.join(modelpath, f'simpleCNN_{args.epochs}epochs_{dt}.h5'))
    with open(os.path.join(modelpath, f'simpleCNN_history_{args.epochs}epochs_{dt}.pickle'), 'wb') as f:
        pickle.dump(hist.history, f, protocol=-1)
