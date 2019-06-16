import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from utils import *
from constants import *


def kfold_lightgbm(x, y, lgb_params, num_folds=5):
    import lightgbm as lgb
    print("Starting lightgbm kfold...")
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(x.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        d_train = lgb.Dataset(x_train, label=y_train)
        model = lgb.train(
            lgb_params,
            train_set=d_train,
            valid_sets=[d_train],
            valid_names=["train"],
            verbose_eval=100,
        )
        model.save_model(f"{LGB_MODELS}/tree_{n_fold}.model")
        oof_preds[valid_idx] = model.predict(x_valid)
        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(y_valid, oof_preds[valid_idx])))

    cv_score = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % cv_score)


def train_lgb(feature_folder, label_file, validate=0.3, allow_cached=True):
    features = combine_csvs(feature_folder)
    combined = preprocess(features, allow_cached=True)
    labels = pd.read_csv(label_file)
    labels = clean_labels(labels)
    combined = combined.merge(labels, on="bookingID", how="left")

    print("columns:", combined.columns)
    x = combined.drop(['label'], axis=1)
    y = combined.label

    params = {
        "num_leaves": 12,
        "objective": "binary",
        "max_depth": 4,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "min_child_weight": 1,
        "lambda_l1": 1,
        "lambda_l2": 0.1,
        # "random_state": 10000019,
        "num_boost_round": 80,
        "metric": "auc",
        "scale_pos_weight": 1,
        "subsample": 0.5
    }

    if validate:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=validate, shuffle=True, random_state=42)
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        model = lgb.train(params, train_set=d_train, valid_sets=[d_train, d_valid], verbose_eval=100)
        model.save_model(f"{LGB_MODELS}/tree0.model")

        y_valid_pred = model.predict(x_valid)
        cv_score = roc_auc_score(y_valid, y_valid_pred)
        print("AUC score:", cv_score)

        # pd.DataFrame({"lgb_preds": y_valid_pred, "label": y_valid, "bookingID": x_valid["bookingID"]}).to_csv("lgb_pred", index=False)

    else:
        kfold_lightgbm(x, y, params, num_folds=5)
        # d_train = lgb.Dataset(x, label=y)
        # model = lgb.train(params, train_set=d_train, valid_sets=[d_train], verbose_eval=100)
        # model.save_model(f"{LGB_MODELS}/tree.model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder', type=str, action='store', dest='feature_folder', required=True)
    parser.add_argument('--label', type=str, action='store', dest='label_file', required=True)
    parser.add_argument('--validate', type=float, action='store', dest='validate', default=0)
    parser.add_argument('--allow_cached', action='store_true')
    # feature_folder = "data/features"
    # label_file = "data/cleaned_labels.csv"
    args = parser.parse_args()
    train_lgb(args.feature_folder, args.label_file, args.validate, args.allow_cached)
