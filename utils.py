import time
import os
from contextlib import contextmanager
import pandas as pd
import numpy as np
from constants import *


@contextmanager
def timer(name):
    print(f"Starting {name}")
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def combine_csvs(feature_folder):
    with timer("Combine CSVs into one"):
        dfs = [pd.read_csv(os.path.join(feature_folder, f)) for f in os.listdir(feature_folder) if f.endswith(".csv")]
        result = pd.concat(dfs, ignore_index=True)
    return result


def decide_label(x):
    if x == 0 or x == 1:
        return x
    if 0 < x < 1:
        return 1
    else:
        raise Exception("Labels should be between 0 and 1")


def clean_labels(labels):
    labels = labels.groupby("bookingID").agg("mean").reset_index()
    labels['label'] = labels['label'].map(decide_label).map(int)
    # labels[['bookingID', 'label']].to_csv(f"{GENERATED_DATA_FOLDER}/cleaned_labels.csv", index=False)
    return labels[['bookingID', 'label']]


def preprocess(features, allow_cached=True):
    output_file = PREPROCESSED_FEATURE
    if allow_cached and os.path.isfile(output_file):
        return pd.read_csv(output_file)

    from tsfresh import extract_features
    fc_parameters = {
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

    kind_to_fc_parameters = {
        "acceleration_abs": fc_parameters,
        "gyro_abs": fc_parameters,
        "Speed": fc_parameters,
        "seconds_copy": {"maximum": None}
    }

    # some feature engineering
    features['acceleration_abs'] = np.sqrt(features['acceleration_x']**2 + features['acceleration_y']**2 + features['acceleration_z']**2)
    features['gyro_abs'] = np.sqrt(features['gyro_x']**2 + features['gyro_y']**2 + features['gyro_z']**2)
    features['seconds_copy'] = features['second']

    with timer("tsfresh extracting features"):
        extracted_features = extract_features(features, column_id="bookingID", column_sort="second", default_fc_parameters={}, kind_to_fc_parameters=kind_to_fc_parameters)

    print(extracted_features.columns)
    print(extracted_features.head())
    extracted_features['bookingID'] = extracted_features.index
    extracted_features.to_csv(output_file, index=False)
    print(f"Extracted features saved to {output_file}")
    return extracted_features