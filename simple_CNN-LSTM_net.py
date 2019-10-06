# %%
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Conv2D, Input, TimeDistributed, Bidirectional, ConvLSTM2D, Dropout, Dense
from tensorflow.keras.models import Sequential

# %%
labelPath = 'bittah-ninja/full_labels.csv'
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


# %%
vidPath = 'fullVidSet'


# %%
def getFrames(filepath):
    cap = cv2.VideoCapture(filepath)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # can try going bigger up to 720*1080 later
            gray = cv2.resize(gray, (224, 224))
            frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    frames = np.array(frames)

    return frames


def formatData(vidPath, df):
    allFrames = []
    for i, row in df.iterrows():
        vid = row.clip_title
        label = row.punch
        filepath = os.path.join(vidPath, vid)
        frames = getFrames(filepath)
        print(frames.shape)
        allFrames.append(frames)


# %%
formatData(vidPath, df)


# %%
allFrames[0]
# %%
np.array(allFrames).shape

# %%
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
          epochs=10,
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
