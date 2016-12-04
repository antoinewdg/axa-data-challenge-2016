import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from featurizer import *
from pandas.tseries.offsets import *
import math
import matplotlib.pyplot as plt
import sys

def learn_structure(in_filename, chunksize=10 ** 6):
    assignments = set()

    dtype = {'ASS_ASSIGNMENT': str}
    cols = ['ASS_ASSIGNMENT']
    chunks = pd.read_csv(in_filename, sep=";", usecols=cols, dtype=dtype, chunksize=chunksize)

    for df in tqdm(chunks):
        assignments.update(df.ASS_ASSIGNMENT.unique())

    return assignments


in_filename = "files/train_groupedby.csv"

assignments_list = list(learn_structure(in_filename))

training_df = load_training_set(in_filename)

measurements = []
training_df = training_df.set_index(['DAY_WE_DS', 'DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS'], drop=False)
for i in trange(0, training_df.shape[0]):
	dic = {}
	(day, datetime, assignment, num_calls) = training_df.index[i]
	time_slot = datetime.time().hour * 2 + (datetime.time().minute / 30)
	month = datetime.date().month
	dic['weekday'] = day
	dic['time_slot'] = time_slot
	dic['month'] = month
	dic['assignment'] = assignment
	dic['num_calls'] = num_calls
	measurements.append(dic)

vec = DictVectorizer()
vec.fit_transform(measurements).toarray()

