import os
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import time
import pdb
from tqdm import tqdm
from keras.models import Model
from keras import optimizers
from keras import layers
from keras.layers import Activation, Flatten, Dense, Dropout, Conv1D, GlobalMaxPooling1D, LeakyReLU, BatchNormalization, Input, ReLU
from keras.layers.merge import concatenate
import keras
from keras.activations import sigmoid, softmax
import tensorflow as tf
import pickle
from keras import regularizers
from keras.utils import to_categorical
import concurrent.futures
from multiprocessing import Process, Queue, current_process, freeze_support
from keras.preprocessing.sequence import pad_sequences


def build_model16(num_samples, num_channels):
    # num_channels = 11
    # num_samples = 20000
    input_timeseries = Input(shape=(num_samples, num_channels,), name='input_timeseries')
    # input_timeseries0 = Input(shape=(num_samples, num_channels,), name='input_timeseries0')
    # input_timeseriese = Input(shape=(num_samples, num_channels,), name='input_timeseriese')
    input_meta = Input(shape=(6,), name='input_meta')
    # _series = concatenate([input_timeseries, input_timeseries0, input_timeseriese])
    _series = concatenate([input_timeseries])
    x = Conv1D(256, 8, padding='same', name='Conv1')(_series)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256, 5, padding='same', name='Conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256, 3, padding='same', name='Conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = GlobalMaxPooling1D()(x)
    x1 = Dense(16, activation='relu', name='dense0')(input_meta)
    x1 = Dense(32, activation='relu', name='dense1')(x1)
    xc = concatenate([x, x1], name='concat')
    x = Dense(256, activation='relu', name='features')(xc)
    out = Activation('sigmoid', name='out')(x)
    # model = Model([input_timeseries, input_timeseries0, input_timeseriese, input_meta], out)
    model = Model([input_timeseries, input_meta], out)

    opt = optimizers.Adam()
    opt.lr = 1e-3
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

def get_ts_array(ts, labels):
    max_len = 1000
    ts = ts[ts["bookingID"].isin(labels["bookingID"])]
    ts = ts.groupby("bookingID").agg(list).reset_index()
    booking_id = ts['bookingID']
    ts = ts.drop(['bookingID'], axis=1)
    arr = np.zeros((ts.shape[0], ts.shape[1], max_len))
    print(ts.columns)
    for i, col in enumerate(ts.columns):
        print(i, col)
        seq = list(ts[col])
        padded_seq = pad_sequences(seq, maxlen=1000, dtype = "float64", padding="pre", truncating="post")
        arr[:,i,:] = padded_seq
    print(arr.shape)


dfs = [pd.read_csv("data/features/" + f) for f in os.listdir("data/features") if f.endswith(".csv")]
features = pd.concat(dfs, ignore_index=True)
labels = pd.read_csv("data/cleaned_labels.csv")

input_ts = get_ts_array(features, labels)


# history=model.fit([sr,sr0,srv,train_meta,train_switch],train_y, batch_size=64,epochs=1,
#                                 validation_data=([validate_timeseries,validate_timeseries0,validate_void,
#                                                   validate_meta,validate_switch],validate_y),
#                                                     verbose=0 )



