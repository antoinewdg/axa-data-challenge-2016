import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from featurizer import *
from pandas.tseries.offsets import *
import math
import matplotlib.pyplot as plt
import sys
from sklearn import tree

training_df = load_training_set("files/train_groupedby.csv")
training_df = training_df.set_index('DATE', drop=False)

submission_df = load_submission("files/submission.txt")
submission_df = submission_df.set_index('DATE', drop=False)

y_true = submission_df.as_matrix()[:, -1]
y_pred = np.zeros(len(y_true))

assignments = list(learn_structure("files/train_groupedby.csv"))

training_df['time_slot'] = training_df.DATE.apply(lambda x: x.time().hour * 2 + (x.time().minute / 30))
training_df['time'] = training_df.index.time
training_df['month'] = training_df.index.month
training_df['weekday'] = pd.DatetimeIndex(training_df['DATE']).weekday
training_df['assignment'] = training_df.ASS_ASSIGNMENT.apply(lambda x: assignments.index(x))

submission_df['time_slot'] = submission_df.DATE.apply(lambda x: x.time().hour * 2 + (x.time().minute / 30))
submission_df['month'] = submission_df.index.month
submission_df['weekday'] = pd.DatetimeIndex(submission_df['DATE']).weekday
submission_df['assignment'] = submission_df.ASS_ASSIGNMENT.apply(lambda x: assignments.index(x))
submission_df2 = submission_df.set_index(['DATE', 'weekday', 'time_slot', 'month', 'ASS_ASSIGNMENT'], drop=False)

submission_df = submission_df.set_index(['DAY_WE_DS', 'DATE', 'ASS_ASSIGNMENT'], drop=False)

X = training_df[['time_slot', 'month', 'weekday', 'assignment']].as_matrix()
y = training_df['CSPL_RECEIVED_CALLS'].as_matrix()
# reg = LinearRegression()
reg = RandomForestRegressor(n_estimators=len(assignments))
# reg = ExtraTreesRegressor(n_estimators=len(assignments))
# reg = BaggingRegressor(n_estimators=len(assignments))
reg.fit(X, y)

importances = reg.feature_importances_

std = np.std([tree.feature_importances_ for tree in reg.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

samples = submission_df[['time_slot', 'month', 'weekday', 'assignment']].as_matrix()
y_pred = reg.predict(samples)

y_pred_round = [int(round(x)) if x > 0 else 0 for x in y_pred]

submission_df.prediction = y_pred_round
submission_df = submission_df.set_index('DATE', drop=False)
submission_df['DATE'] = submission_df.index.strftime('%Y-%m-%d %H:%M:%S.000')
# print submission_df

submission_df[['DATE', 'ASS_ASSIGNMENT', 'prediction']].to_csv('results/submission13.txt', sep='\t', index=False)

print('MSE round: '),
print(mean_squared_error(y_true, y_pred_round))

print('MSE not round: '),
print(mean_squared_error(y_true, y_pred))
