import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import timer

features_folder = "data/features/"

fc_parameters = {
    "length": None,
    "sum_values": None,
    "minimum": None,
    "maximum": None,
    "mean": None,
    "median": None,
    "mean_change": None,
    "mean_abs_change": None,
    "count_above_mean": None,
    "standard_deviation": None,
    "longest_strike_above_mean": None,
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2]],
    # "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
    "number_cwt_peaks": [{"n": n} for n in [1, 5]],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    # "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]]
}

REGENERATE = True

if REGENERATE:
    dfs = [pd.read_csv("data/features/" + f) for f in os.listdir("data/features")]
    features = pd.concat(dfs, ignore_index=True)

    # some feature engineering
    features['acceleration_abs'] = np.sqrt(features['acceleration_x']**2 + features['acceleration_y']**2 + features['acceleration_z']**2)
    features['gyro_abs'] = np.sqrt(features['gyro_x']**2 + features['gyro_y']**2 + features['gyro_z']**2)    


    with timer("tsfresh extracting features"):
        extracted_features = extract_features(features.drop(['Accuracy'], axis=1), column_id="bookingID", column_sort="second", default_fc_parameters = fc_parameters)

    # print("len1", len(extracted_features))
    # # engineer some aggregate features
    # custom_agg_features = features.groupby("bookingID").agg({"second": np.max}).reset_index().rename(columns={"second": "trip_duration"})
    # print("len2", len(custom_agg_features))

    # # merge them in
    # extracted_features = extracted_features.merge(custom_agg_features, left_index=True, right_on="bookingID", how="left")
    # print("len3", len(extracted_features))

    # clean off some useless features
    # extracted_features = extracted_features.drop(["Accuracy__maximum", "Accuracy__minimum", "Accuracy__standard_deviation", "Accuracy__sum_values"], axis=1)


    labels = pd.read_csv("data/cleaned_labels.csv")
    combined = extracted_features.merge(labels, left_index=True, right_on="bookingID", how="left")
    print("len4", len(combined))
    combined.to_csv("generated_data/tsfresh_features_v3.csv", index=False)

combined = pd.read_csv("generated_data/tsfresh_features_v3.csv")

print("columns:", combined.columns)
x = combined.drop(['bookingID', 'label'], axis=1)
y = combined.label.values

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, shuffle=True, random_state=42)


############### Linear models
# from sklearn.linear_model import Ridge, Lasso

# model = Lasso(alpha=0.001)
# model.fit(x_train, y_train)
# y_valid_pred = model.predict(x_valid)
#################

############### lgb
params = {
    "num_leaves": 16,
    "objective": "binary",
    "max_depth": 5,
    "learning_rate": 0.003,
    "boosting_type": "gbdt",
    # "feature_fraction": 0.7,
    # "bagging_fraction": 0.7,
    # "bagging_freq": 1,
    # "lambda_l1": 0.1,
    # "lambda_l2": 0.1,
    "random_state": 10000019,
    # "verbosity": 1,
    "num_boost_round": 1000,
    "metric": "auc",
    # "scale_pos_weight": 4,
}

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)
model = lgb.train(params, train_set=d_train, valid_sets=[d_valid], verbose_eval=100)
y_valid_pred = model.predict(x_valid)
###########################

print(y_valid_pred)
cv_score = roc_auc_score(y_valid, y_valid_pred)
print(cv_score)

rounded_y_valid_preds = np.round(np.clip(y_valid_pred, 0, 1))
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_valid, rounded_y_valid_preds)
print("accuracy: ", acc)