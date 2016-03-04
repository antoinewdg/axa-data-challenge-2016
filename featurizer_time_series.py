import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from clean_training_set import load_clean_training_set
from featurizer import *
from pandas.tseries.offsets import *
from learn_structure import load_structure
import math
import os, pickle
import datetime
from collections import deque

_n_features = 10


def _update_data(df, old_values):
    for index, row in df.iterrows():
        triplet = (row.DAY_WE_DS, row.ASS_ASSIGNMENT, row.DATE.time())
        if triplet not in old_values:
            old_values[triplet] = deque([], maxlen=_n_features)
        if len(old_values[triplet]) >= _n_features:
            old_values[triplet].popleft()
        old_values[triplet].append(row.CSPL_RECEIVED_CALLS)


def _deque_to_features(deq):
    res = list(deq)
    while len(res) < 10:
        res.append(np.nan)
    return res


def featurize_time_series(training_df, df, features):
    print("Featurizing time series")

    min_date = df.DATE.min()
    max_date = df.DATE.max()
    old_df = training_df[(training_df.DATE < min_date - DateOffset(days=3))]

    old_values = {}
    _update_data(old_df, old_values)

    prev_date = min_date
    current_date = min_date + DateOffset(days=7)

    indices = np.zeros(len(df))
    new_features = np.full((len(df), _n_features), np.nan)

    i = 0
    with tqdm(total=int((max_date - min_date).days / 7) + 1) as pbar:
        while prev_date < max_date:
            sub_df = df[(df.DATE >= prev_date) & (df.DATE < current_date)]
            for index, row in sub_df.iterrows():
                triplet = (row.DAY_WE_DS, row.ASS_ASSIGNMENT, row.DATE.time())
                if triplet in old_values:
                    indices[i] = index
                    new_features[i] = _deque_to_features(old_values[triplet])
                    i += 1

            old_df = training_df[(training_df.DATE >= prev_date - DateOffset(days=3)) &
                                 (training_df.DATE < current_date - DateOffset(days=3))]
            _update_data(old_df, old_values)
            prev_date = current_date
            current_date = prev_date + DateOffset(days=7)
            pbar.update(1)

    dict = {}
    for j in range(new_features.shape[1]):
        dict['prev_value_' + str(j)] = new_features[:, j]

    new_df = pd.DataFrame(dict, index=indices)
    return features.join(new_df)


def featurize_time_series_submission(submission_df, features, structure):
    print("Featurizing time series")
    print(features.shape[1])
    assignments = structure['ASS_ASSIGNMENT']
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    ass_dfs = {}
    for day in days:
        for ass in assignments:
            ass_dfs[day + '_' + ass] = pd.read_pickle('files/split/' + day + '_' + ass + '.pkl')

    new_features = np.full((len(submission_df), _n_features), np.nan)

    submission_df = submission_df.set_index(['DAY_WE_DS', 'DATE', 'ASS_ASSIGNMENT'], drop=False)
    for i in trange(0, submission_df.shape[0]):
        (day, datetime, ass) = submission_df.index[i]
        df = ass_dfs[day + '_' + ass][(ass_dfs[day + '_' + ass].DATE.dt.hour == datetime.hour) &
                                      (ass_dfs[day + '_' + ass].DATE.dt.minute == datetime.minute)]
        df = df[df.DATE < datetime - DateOffset(days=3)]
        old_values = df.tail(_n_features)['CSPL_RECEIVED_CALLS'].as_matrix()

        # print(old_values)
        for j in range(len(old_values)):
            new_features[i, j] = old_values[j]

    for j in range(_n_features):
        features['prev_value_' + str(j)] = new_features[:, j]

    return features
