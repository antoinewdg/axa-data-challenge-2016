
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

def learn_structure(filename, chunksize=10 ** 6):
    assignments = set()

    dtype = {'ASS_ASSIGNMENT': str}
    cols = ['ASS_ASSIGNMENT']
    chunks = pd.read_csv(filename, sep=";", usecols=cols, dtype=dtype, chunksize=chunksize)

    for df in tqdm(chunks):
        assignments.update(df.ASS_ASSIGNMENT.unique())

    print(assignments)

    return assignments

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
            features['time_slot_' + str(2 * h + s)] = ((df.DATE.dt.hour == h) & (df.DATE.dt.minute == 30 * s)).astype(int)

    print ()

def featurize_assignment(df, features, assignments):
    print("Featurizing assignment")
    for assignment in assignments:
        features[assignment] = (df.ASS_ASSIGNMENT == assignment).astype(int)

    print ()

def featurize_number_of_calls(df, features):
	print("Featurizing assignment")
	features['n_calls'] = df.CSPL_RECEIVED_CALLS
	print ()

def featurize_weekend(df, features):
	print("Featurizing weekend")
	features['n_calls'] = df.CSPL_RECEIVED_CALLS
	print ()


dtype = {
            'DATE': object,
            'WEEK_END': int,
            'DAY_WE_DS': str,
            'ASS_ASSIGNMENT': str,
            'CSPL_RECEIVED_CALLS': int
        }

cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
chunks = pd.read_csv("../data/train_2011_2012.csv", sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'], chunksize=10**6)

df = pd.DataFrame()
for chunk in chunks:
    aux = chunk.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)['CSPL_RECEIVED_CALLS'].sum()
    df = pd.concat([df, aux])

df = df.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)['CSPL_RECEIVED_CALLS'].sum()



# # Featurize database

# In[56]:

features = pd.DataFrame()
featurize_day_of_the_week(df,features)
featurize_time_slot(df, features)
assignments = learn_structure("../data/train_2011_2012.csv")
featurize_assignment(df, features, assignments)
featurize_number_of_calls(df, features)

features['DATE'] = df.DATE

print features.head(3)


# # Submission

# In[57]:

dtype = {
            'DATE': object,
            'ASS_ASSIGNMENT': str,
            'prediction': int
        }

cols = ['DATE', 'ASS_ASSIGNMENT', 'prediction']
df2 = pd.read_csv("../data/submission_test.txt", sep="\t", usecols=cols, dtype=dtype, parse_dates=['DATE'])

#Get the answer vector
y_true = np.asarray(df2['prediction'])



#Featurize the submission file
submission_features = pd.DataFrame()

weekdays = pd.DatetimeIndex(df2['DATE']).weekday
days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

for day in range(5,7):
    submission_features['WEEK_END'] = (weekdays == day).astype(int)
    
for day in range(7):
    submission_features[days[day]] = (weekdays == day).astype(int)



featurize_time_slot(df2, submission_features)
featurize_assignment(df2, submission_features, assignments)


# # Prediction

# In[58]:

X_test = np.asarray(submission_features)
clf = SGDRegressor()
y_pred = np.zeros(len(X_test))

local_df = features[features.DATE < df2.DATE[0] - DateOffset(days = 3)]

X_train = np.asarray(local_df)[:, :-2]
y_train = np.asarray(local_df)[:, -2]

clf.partial_fit(X_train, y_train)
y_pred[0] = clf.predict(X_test[0])

for i in trange(1,len(X_test)):
    local_df = features[(features.DATE > df2.DATE[i-1]) & (features.DATE < (df2.DATE[i] - DateOffset(days = 3)))]
    X_train = np.asarray(local_df)[:, :-2]
    y_train = np.asarray(local_df)[:, -2]
    if X_train.shape[0] != 0:
	    clf.partial_fit(X_train, y_train)
    y_pred[i] = clf.predict(X_test[i])


# In[59]:

y_pred_round = [int(math.ceil(x)) if x > 0 else 0 for x in y_pred]
#print(y_pred_round)


# # Output

# In[62]:

df2.prediction = pd.Series(y_pred_round)
df2.to_csv('../results/submission_out.txt', sep='\t', index=False)


print('MSE round: '),
print(mean_squared_error(y_true, y_pred_round))


print('MSE not round: '),
print(mean_squared_error(y_true, y_pred))


# In[ ]:



