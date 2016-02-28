
# coding: utf-8

# # Read data

# In[75]:

import pandas as pd

dtype = {
            'DATE': object,
            'WEEK_END': int,
            'DAY_WE_DS': str,
            'ASS_ASSIGNMENT': str,
            'CSPL_RECEIVED_CALLS': int
        }

cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
chunks = pd.read_csv("files/train_2011_2012.csv", sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'], chunksize=10**6)


# # Reduce data

# In[76]:

df = pd.DataFrame()
for chunk in chunks:
    aux = chunk.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)['CSPL_RECEIVED_CALLS'].sum()
    df = pd.concat([df, aux])
    # print(df.head())


df = df.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)['CSPL_RECEIVED_CALLS'].sum()
# # Test

# In[84]:

from featurizer import *
import numpy as np

feat = Featurizer()
features = feat._featurize_chunk(df)
train_size = int(0.98*df.shape[0])
X_train = np.asarray(features)[0:train_size, :-1]
y_train = np.asarray(features)[0:train_size, -1]

X_test = np.asarray(features)[train_size:, :-1]
y_test = np.asarray(features)[train_size:, -1]


# In[85]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(mean_squared_error(y_pred, y_test))


# Submission
dtype = {
            'DATE': object,
            'ASS_ASSIGNMENT': str,
            'prediction': int
        }

cols = ['DATE', 'ASS_ASSIGNMENT', 'prediction']
chunks = pd.read_csv("files/submission.txt", sep="\t", usecols=cols, dtype=dtype, parse_dates=['DATE'])

print chunks





