import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import *
import math


def learn_structure(filename, chunksize=10 ** 6):
    assignments = set()

    dtype = {'ASS_ASSIGNMENT': str}
    cols = ['ASS_ASSIGNMENT']
    chunks = pd.read_csv(filename, sep=";", usecols=cols, dtype=dtype, chunksize=chunksize)

    for df in tqdm(chunks):
        assignments.update(df.ASS_ASSIGNMENT.unique())

    print(assignments)

    return assignments


def load_training_set(filename):
    dtype = {
        'DATE': object,
        'WEEK_END': int,
        'DAY_WE_DS': str,
        'ASS_ASSIGNMENT': str,
        'CSPL_RECEIVED_CALLS': int
    }

    cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
    chunks = pd.read_csv(filename, sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'],
                         chunksize=10 ** 6)

    df = pd.DataFrame()
    for chunk in chunks:
        aux = chunk.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)[
            'CSPL_RECEIVED_CALLS'].sum()
        df = pd.concat([df, aux])

    df = df.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)[
        'CSPL_RECEIVED_CALLS'].sum()

    return df


def load_submission(filename):
    dtype = {
        'DATE': object,
        'ASS_ASSIGNMENT': str,
        'prediction': int
    }

    cols = ['DATE', 'ASS_ASSIGNMENT', 'prediction']
    df = pd.read_csv(filename, sep="\t", usecols=cols, dtype=dtype, parse_dates=['DATE'])

    weekdays = pd.DatetimeIndex(df['DATE']).weekday
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    df['WEEK_END'] = ((weekdays == 5) | (weekdays == 6)).astype(int)
    df['DAY_WE_DS'] = [days[w] for w in weekdays]
    df['CSPL_RECEIVED_CALLS'] = df['prediction']
    return df


def featurize_all(df, assignments):
    features = pd.DataFrame()
    featurize_day_of_the_week(df, features)
    featurize_time_slot(df, features)
    featurize_assignment(df, features, assignments)
    featurize_number_of_calls(df, features)
    return features


def featurize_day_of_the_week(df, features):
    print("Featurizing days of the week")

    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    features['is_week_end'] = df.WEEK_END
    for day in days:
        features[day] = (df.DAY_WE_DS == day).astype(int)

    print()


def featurize_time_slot(df, features):
    print("Featurizing time slots")

    for h in trange(24):
        for s in range(2):
            features['time_slot_' + str(2 * h + s)] = ((df.DATE.dt.hour == h) & (df.DATE.dt.minute == 30 * s)).astype(
                int)

    print()


def featurize_assignment(df, features, assignments):
    print("Featurizing assignment")
    for assignment in assignments:
        features[assignment] = (df.ASS_ASSIGNMENT == assignment).astype(int)

    print()


def featurize_number_of_calls(df, features):
    print("Featurizing assignment")
    features['n_calls'] = df.CSPL_RECEIVED_CALLS
    print()


def featurize_weekend(df, features):
    print("Featurizing weekend")
    features['n_calls'] = df.CSPL_RECEIVED_CALLS
    print()
