import numpy as np
import pandas as pd
import sys

from featurizer import *
from learn_structure import load_structure
from clean_training_set import load_clean_training_set


def load_featurized_training_set(filename):
    return pd.read_pickle(filename)


def featurize_training_set(in_filename, out_filename):
    df = load_clean_training_set(in_filename)
    structure = load_structure()
    features = featurize_all(df, structure['ASS_ASSIGNMENT'])
    features['DATE'] = df.DATE

    features.to_pickle(out_filename)


if __name__ == "__main__":
    featurize_training_set(sys.argv[1], sys.argv[2])
    features = load_featurized_training_set(sys.argv[2])
    print(features.head(3))
