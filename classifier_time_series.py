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

training_df = load_training_set("files/train_groupedby.csv")
training_df = training_df.set_index('DATE', drop=False)

submission_df = load_submission("files/submission.txt")
submission_df = submission_df.set_index('DATE', drop=False)

y_true = submission_df.as_matrix()[:, -1]
y_pred = np.zeros(len(y_true))

for i in trange(0, submission_df.shape[0]):
	submission_df = submission_df.set_index(['DAY_WE_DS', 'DATE', 'ASS_ASSIGNMENT'], drop=False)
	(day, datetime, assignment) = submission_df.index[i]
	time = datetime.time()

	local_df = training_df[training_df.ASS_ASSIGNMENT == assignment]
	local_df = local_df[local_df.DAY_WE_DS == day]
	local_df = local_df[local_df.index.time == time]

	y_pred[i] = local_df['CSPL_RECEIVED_CALLS'].mean()
	# print y_pred[i]

y_pred_round = [int(math.ceil(x)) if x > 0 else 0 for x in y_pred]
# print(y_pred_round)

submission_df.prediction = y_pred_round
print submission_df
submission_df[['DATE', 'ASS_ASSIGNMENT', 'prediction']].to_csv('results/submission.txt', sep='\t', index=False)

print('MSE round: '),
print(mean_squared_error(y_true, y_pred_round))

print('MSE not round: '),
print(mean_squared_error(y_true, y_pred))
