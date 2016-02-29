
# coding: utf-8

# # Initialization

# In[55]:

import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import *
import math

dtype = {
            'DATE': object,
            'ASS_ASSIGNMENT': str,
            'CSPL_RECEIVED_CALLS': int
        }

cols = ['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
chunks = pd.read_csv("../data/train_2011_2012.csv", sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'], chunksize=10**6)

df = pd.DataFrame()
for chunk in chunks:
    aux = chunk.groupby(['DATE', 'ASS_ASSIGNMENT'], as_index=False, sort=True)['CSPL_RECEIVED_CALLS'].sum()
    df = pd.concat([df, aux])

df = df.groupby(['DATE', 'ASS_ASSIGNMENT'], as_index=False, sort=True)['CSPL_RECEIVED_CALLS'].sum()

df['prediction'] = df['CSPL_RECEIVED_CALLS']

num_rows = 12408

df['DATE'] = pd.to_datetime(df['DATE'].astype(str))
df = df[df['DATE'] > '2011-12-31'] 
df = df.sample(n=num_rows)
df = df.sort_values(by=['DATE'])

<<<<<<< HEAD
df.to_csv('files/submission_test.txt', sep='\t', index=False, columns=['DATE', 'ASS_ASSIGNMENT', 'prediction'])



=======
df.to_csv('../data/submission_test.txt', sep='\t', index=False, columns=['DATE', 'ASS_ASSIGNMENT', 'prediction'])
>>>>>>> 475fedb4f3d686be75314218949c4585ee612b42
