import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from lgbm import preprocess
from nn import prepare_nn_input
from constants import *
from utils import *


def predict(feature_folder):
    print("Begin preprocessing...")
    features = combine_csvs(feature_folder)
    # meta = preprocess(features)
    meta = pd.read_csv(f"{GENERATED_DATA_FOLDER}/tsfresh_features_v5.csv")
    booking_id = meta['bookingID']
    lgb_preds = []
    nn_preds = []

    # lgb predictions
    lgb_files = [x for x in os.listdir(LGB_MODELS) if x.endswith(".model")]
    print("Start predicting LGB")
    for f in tqdm(lgb_files):
        model = lgb.Booster(model_file=os.path.join(LGB_MODELS, f))
        lgb_preds.append(model.predict(meta))

    # cnn predictions
    print("Start preparing nn input")
    inputs = prepare_nn_input(features, meta)
    cnn_files = [x for x in os.listdir(CNN_MODELS) if x.endswith(".h5")]
    print("Start predicting NN")
    for f in tqdm(cnn_files):
        model = load_model(os.path.join(CNN_MODELS, f))
        nn_preds.append(model.predict(inputs))

    lgb_all = np.mean(lgb_preds, axis=0)
    nn_all = np.mean(nn_preds, axis=0).transpose((1, 0))[0]
    y_preds = lgb_all * 0.7 + nn_all * 0.3
    return pd.DataFrame({"bookingID": booking_id, "p": y_preds})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder', type=str, action='store', dest='feature_folder', required=True, help='folder where the time series csvs are stored')
    parser.add_argument('--label', type=str, action='store', dest='label_file', help='optional, csv file of the labels')
    args = parser.parse_args()

    preds = predict(args.feature_folder)
    preds.to_csv("output.csv", index=False)
    print("Saved predictions to output.csv")
    if args.label_file:
        labels = pd.read_csv(args.label_file)
        y_true = labels.merge(preds, on="bookingID", how="left")['label']
        score = roc_auc_score(y_true, preds['p'])
        print("AUC", score)
