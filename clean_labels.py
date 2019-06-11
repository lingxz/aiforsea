import pandas as pd

labels = pd.read_csv("data/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
labels = labels.groupby("bookingID").agg("mean").reset_index()

def decide_label(x):
    if x == 0 or x == 1:
        return x
    if 0 < x < 1:
        return 1
    else:
        raise Exception("should not have this")

labels['label'] = labels['label'].map(decide_label).map(int)
labels[['bookingID', 'label']].to_csv("data/cleaned_labels.csv", index=False)
