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


def update_data(df, sums, numbers):
    for index, row in df.iterrows():
        triplet = (row.DAY_WE_DS, row.ASS_ASSIGNMENT, row.DATE.time())
        if triplet not in sums:
            sums[triplet] = 0
            numbers[triplet] = 0
        sums[triplet] += row.CSPL_RECEIVED_CALLS
        numbers[triplet] += 1


def load_data(df):
    sums = {}
    numbers = {}

    update_data(df, sums, numbers)

    pickle.dump((sums, numbers), open("files/time_series_data.pkl", "wb"))
    return sums, numbers


training_df = load_clean_training_set("files/train_clean.pkl")

submission_df = load_submission("files/submission_test.txt")

y_true = np.copy(submission_df['prediction'].as_matrix())
y_pred = np.zeros(len(y_true))

min_date = submission_df['DATE'].min()
max_date = submission_df['DATE'].max()

old_df = training_df[training_df.DATE < min_date - DateOffset(days=3)]

sums, numbers = load_data(old_df)

prev_date = min_date
current_date = min_date + DateOffset(days=7)

n = 0
with tqdm(total=((max_date - min_date).days / 7) + 1) as pbar:
    while prev_date < max_date:
        sub_df = submission_df[(submission_df.DATE >= prev_date) & (submission_df.DATE < current_date)]
        for index, row in sub_df.iterrows():
            triplet = (row.DAY_WE_DS, row.ASS_ASSIGNMENT, row.DATE.time())
            if triplet in sums:
                y_pred[index] = float(sums[triplet]) / numbers[triplet]

        old_df = training_df[(training_df.DATE >= prev_date - DateOffset(days=3)) &
                             (training_df.DATE < current_date - DateOffset(days=3))]
        update_data(old_df, sums, numbers)
        prev_date = current_date
        current_date = prev_date + DateOffset(days=7)
        pbar.update(1)

print(n)
y_pred_round = [int(math.ceil(x)) if x > 0 else 0 for x in y_pred]
submission_df.prediction = y_pred_round
print(submission_df)
submission_df.DATE = submission_df.DATE.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
submission_df[['DATE', 'ASS_ASSIGNMENT', 'prediction']].to_csv('results/submission_real.txt', sep='\t', index=False)
#
print('MSE round: '),
print(mean_squared_error(y_true, y_pred_round))
#
print('MSE not round: '),
print(mean_squared_error(y_true, y_pred))
