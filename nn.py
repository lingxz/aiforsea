import os
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
from keras import optimizers
from keras import layers
from keras.layers import Activation, Flatten, Dense, Dropout, Conv1D, GlobalMaxPooling1D, MaxPooling1D, LeakyReLU, BatchNormalization, Input, ReLU, LSTM
from keras.layers.merge import concatenate
import keras
from keras.activations import sigmoid
from keras import regularizers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from multiprocessing import Process, Queue, current_process, freeze_support
from multiprocessing.pool import ThreadPool, Pool
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from utils import *
from constants import *

################# for GPU ####################
import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
##############################################

params = {
    "max_len": 1000,
    "epochs": 5,
    "starting_batch_size": 5,
    "adam_lr": 1e-3,
    "num_nns": 16
}

def build_model16(timeseries_length, num_channels, num_meta):
    input_timeseries = Input(shape=(timeseries_length, num_channels,), name='input_timeseries')
    input_meta = Input(shape=(num_meta,), name='input_meta')
    x = Conv1D(128, 5, activation='relu')(input_timeseries)
    x = MaxPooling1D()(x)
    x = Conv1D(256, 5, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Conv1D(32, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x1 = Dense(16, activation='relu', name='dense0')(input_meta)
    x1 = Dense(32, activation='relu', name='dense1')(x1)
    xc = concatenate([x, x1], name='concat')
    x = Dense(64, activation='relu', name='features')(xc)
    out = Dense(1, activation='sigmoid', name='out')(x)
    model = Model([input_timeseries, input_meta], out)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=params['adam_lr']))
    return model


def get_ts_array(ts, meta):
    max_len = params["max_len"]
    ts = ts.sort_values(by="second", ascending=True).groupby("bookingID").agg(list).reset_index()
    ts = meta.merge(ts, on="bookingID", how="left")

    booking_id = ts['bookingID']
    arr_meta = ts[meta.columns].values
    ts = ts.drop(['bookingID'] + list(meta.columns), axis=1)

    # normalize meta info
    meta_scaler = StandardScaler()
    arr_meta = meta_scaler.fit_transform(arr_meta)

    arr = np.zeros((ts.shape[0], ts.shape[1], max_len))
    with timer("prepare and pad time series"):
        for i, col in enumerate(ts.columns):
            padded_seq = pad_sequences(list(ts[col]), maxlen=max_len, dtype = "float64", padding="post", truncating="post")
            ts_scaler = StandardScaler()
            padded_seq = ts_scaler.fit_transform(padded_seq)
            arr[:,i,:] = padded_seq

    return arr.astype('float32'), arr_meta.astype('float32'), booking_id

def fit_model(index, input_train, y_train, num_channels, num_meta):
    model = build_model16(params['max_len'], num_channels, num_meta)
    for i in range(params['epochs']):
        model.fit(
            input_train,
            y_train,
            batch_size = 2 ** (params["starting_batch_size"] + i),
            epochs=1,
            verbose=1
        )
    model.save(f"{CNN_MODELS}/cnn_{index}.h5")
    return model


def prepare_nn_input(features, meta):
    input_ts, input_meta, booking_id = get_ts_array(features, meta)
    return [np.transpose(input_ts, (0, 2, 1)), input_meta]


def train_nn(feature_folder, label_file, validate=0.33, allow_cached=True):
    feature_folder = "data/features"
    label_file = "data/cleaned_labels.csv"
    features = combine_csvs(feature_folder)
    labels = pd.read_csv(label_file)  # TODO: deduplicate
    labels = clean_labels(labels)

    meta = preprocess(features, allow_cached=allow_cached)
    input_ts, input_meta, booking_id = get_ts_array(features, meta)
    y = pd.DataFrame({"bookingID": booking_id}).merge(labels, on="bookingID", how="left").label.values

    if validate:
        input_ts_train, input_ts_test, input_meta_train, input_meta_test, booking_id_train, booking_id_test, y_train, y_test = train_test_split(input_ts, input_meta, booking_id, y, test_size=validate, shuffle=True, random_state=42)

        input_train = [np.transpose(input_ts_train, (0, 2, 1)), input_meta_train]
        input_test = [np.transpose(input_ts_test, (0, 2, 1)), input_meta_test]

        preds = []
        with timer("NN training"):
            for i in range(params["num_nns"]):
                model = fit_model(i, input_train, y_train, input_ts.shape[1], input_meta.shape[1])
                preds.append(model.predict(input_test))

        y_preds = np.mean(preds, axis=0).transpose((1, 0))[0]
        print("AUC score", roc_auc_score(y_test, y_preds))
        # pd.DataFrame({"bookingID": booking_id_test, "nn_preds": y_preds, "nn_label": y_test}).to_csv("nn_pred", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder', type=str, action='store', dest='feature_folder', required=True, help='folder where the time series csvs are stored')
    parser.add_argument('--label', type=str, action='store', dest='label_file', required=True, help='csv file of the labels')
    parser.add_argument('--validate', type=float, action='store', dest='validate', default=0, help='optional, fraction of the training set that you want to validate on (default: 0)')
    parser.add_argument('--allow_cached', action='store_true', help='optional, allow to use cached tsfresh features to avoid recalculating (default: false)')
    args = parser.parse_args()

    train_nn(args.feature_folder, args.label_file, args.validate, args.allow_cached)
