import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from featurizer import *
from pandas.tseries.offsets import *
import math


in_filename = "files/train_groupedby.csv"
out_filename_prefix = "files/assign/"
df = load_training_set(in_filename)

assignments = learn_structure(in_filename)

for ass in assignments:
	local_df = df[df.ASS_ASSIGNMENT == ass]
	local_df.to_csv(out_filename_prefix + 'train_' + str(ass) + '.csv', sep=';', index=False)
