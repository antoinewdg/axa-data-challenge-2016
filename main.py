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
from featurizer import *
from pandas.tseries.offsets import *
import math

df = load_training_set("files/train_france.csv")
assignments = learn_structure("files/train_france.csv")
features = featurize_all(df, assignments)
features['DATE'] = df.DATE

print(features.head(3))

# # Submission

# In[57]:

df2 = load_submission("files/submission.txt")
submission_features = featurize_all(df2, assignments)
featurize_day_of_the_week(df2, submission_features)

# # Prediction

# In[58]:

X_test = np.asarray(submission_features)
clf = SGDRegressor()
y_pred = np.zeros(len(X_test))

local_df = features[features.DATE < df2.DATE[0] - DateOffset(days=3)]

X_train = np.asarray(local_df)[:, :-2]
y_train = np.asarray(local_df)[:, -2]

clf.partial_fit(X_train, y_train)
y_pred[0] = clf.predict(X_test[0])

for i in trange(1, len(X_test)):
    local_df = features[(features.DATE > df2.DATE[i - 1]) & (features.DATE < (df2.DATE[i] - DateOffset(days=3)))]
    X_train = np.asarray(local_df)[:, :-2]
    y_train = np.asarray(local_df)[:, -2]
    if X_train.shape[0] != 0:
        clf.partial_fit(X_train, y_train)
    y_pred[i] = clf.predict([X_test[i]])[0]

# In[59]:

y_pred_round = [int(math.ceil(x)) if x > 0 else 0 for x in y_pred]
print(y_pred_round)

# # Output

# In[62]:

df2.prediction = pd.Series(y_pred_round)
df2.to_csv('submission_out.txt', sep='\t', index=False)


# In[ ]:
