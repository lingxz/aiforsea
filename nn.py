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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import timer

MAX_LEN = 1000

def build_model16(num_samples, num_channels, num_meta):
    # num_channels = 10
    # num_samples = 1000
    input_timeseries = Input(shape=(num_samples, num_channels,), name='input_timeseries')
    # input_timeseries0 = Input(shape=(num_samples, num_channels,), name='input_timeseries0')
    # input_timeseriese = Input(shape=(num_samples, num_channels,), name='input_timeseriese')
    input_meta = Input(shape=(num_meta,), name='input_meta')
    # _series = concatenate([input_timeseries, input_timeseries0, input_timeseriese])
    x = Conv1D(256, 8, padding='same', name='Conv1')(input_timeseries)
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
    out = Dense(1, activation='sigmoid', name='out')(x)
    # out = Activation('sigmoid', name='out')(x)
    # model = Model([input_timeseries, input_timeseries0, input_timeseriese, input_meta], out)
    model = Model([input_timeseries, input_meta], out)

    opt = optimizers.Adam()
    opt.lr = 1e-3
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

def get_ts_array(ts, meta, labels):
    max_len = MAX_LEN
    # ts = ts[ts["bookingID"].isin(labels["bookingID"])]
    ts = ts.groupby("bookingID").agg(list).reset_index()
    ts = ts.merge(labels, on="bookingID", how="inner").merge(meta, on="bookingID", how="left")
    print(ts.columns)
    # ts = labels.merge(ts, on="bookingID", how="left").merge(meta, on="bookingID", how="left")

    booking_id = ts['bookingID']
    labels = ts['label']
    arr_meta = ts[meta.columns].values
    ts = ts.drop(['bookingID', 'label'] + list(meta.columns), axis=1)

    # normalize meta
    meta_scaler = StandardScaler()
    arr_meta = meta_scaler.fit_transform(arr_meta)

    arr = np.zeros((ts.shape[0], ts.shape[1], max_len))
    print(ts.columns)
    for i, col in enumerate(ts.columns):
        print(i, col)
        seq = list(ts[col])
        padded_seq = pad_sequences(seq, maxlen=1000, dtype = "float64", padding="pre", truncating="post")
        ts_scaler = StandardScaler()
        padded_seq = ts_scaler.fit_transform(padded_seq)
        arr[:,i,:] = padded_seq
    print(arr.shape)
    return arr, arr_meta, booking_id, labels


dfs = [pd.read_csv("data/features/" + f) for f in os.listdir("data/features") if f.endswith(".csv")]
features = pd.concat(dfs, ignore_index=True)
labels = pd.read_csv("data/cleaned_labels.csv")
combined = pd.read_csv("generated_data/tsfresh_features_v4.csv")

input_ts, input_meta, booking_id, y = get_ts_array(features, combined.drop(['label'], axis=1), labels)
input_ts_train, input_ts_test, input_meta_train, input_meta_test, booking_id_train, booking_id_test, y_train, y_test = train_test_split(input_ts, input_meta, booking_id, y, test_size=0.33, shuffle=True, random_state=42)

print("input_ts shape", input_ts.shape)

y_preds_all = np.zeros(len(y_test))
for _ in range(8):
    model = build_model16(MAX_LEN, input_ts.shape[1], input_meta.shape[1])
    # with timer("NN training"):
    model.fit([np.transpose(input_ts_train, (0, 2, 1)), input_meta_train], y_train, batch_size=64, epochs=3, verbose=1)

    # with timer("NN Predict"):
    y_preds = model.predict([np.transpose(input_ts_test, (0, 2, 1)), input_meta_test], batch_size=64)
    y_preds_all += y_preds/8.


from sklearn.metrics import roc_auc_score
cv_score = roc_auc_score(y_test, y_preds_all)
print(cv_score)

