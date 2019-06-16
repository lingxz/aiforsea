# Grab AI for SEA (Safety)

This solution blends 2 kinds of models: 5 fold LightGBM and 16 bagged CNNs. The resulting AUC score is about 0.74 - 0.75, depending on the train test split. 

- Some basic preprocessing removes duplicates and adds some useful timeseries (such as the magnitude of acceleration). 
- The LGB model uses time series aggregation features generated by [`tsfresh`](https://tsfresh.readthedocs.io). These include (but are not limited to) max acceleration, acceleration std, etc. 
- The CNN is a Fully 1D Convolutional Neural Network followed by a GlobalMaxPooling, similar to the one described here [Time Series Classiﬁcation from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/pdf/1611.06455.pdf). This is concatenated with a small MLP that takes in the tsfresh features used in the LGB.  

## Requirements

python>=3.6

## How to predict

The models are pretrained. To predict, do:

```sh
python predict.py --feature_folder="data/features" --label="data/label_file.csv"
```

`--feature_folder` is the folder where all the time series data is stored, and `--label` is an optional field to input the true labels for scoring the predictions. If this is not present, the predictions will just be saved to an output file `output.csv`. 

## How to train

To train the LGB, run

```sh
python lgbm.py --feature_folder="data/features" --label="data/label_file.csv" --validate=0.2
```

Similarly, to train the CNN (takes a few hours without a gpu):

```sh
python nn.py --feature_folder="data/features" --label="data/label_file.csv" --validate=0.2
```


Command line options (same for `nn.py` and `lgbm.py`):

```
usage: nn.py [-h] --feature_folder FEATURE_FOLDER --label LABEL_FILE
             [--validate VALIDATE] [--allow_cached]

optional arguments:
  -h, --help            show this help message and exit
  --feature_folder FEATURE_FOLDER
                        folder where the time series csvs are stored
  --label LABEL_FILE    csv file of the labels
  --validate VALIDATE   optional, fraction of the training set that you want
                        to validate on (default: 0)
  --allow_cached        optional, allow to use cached tsfresh features to
                        avoid recalculating (default: false)
```

