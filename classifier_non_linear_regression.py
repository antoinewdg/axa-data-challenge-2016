import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from featurizer import *
from pandas.tseries.offsets import *
import math
import matplotlib.pyplot as plt
import sys

def execute(n):
	training_df = load_training_set("files/train_groupedby.csv")
	training_df = training_df.set_index('DATE', drop=False)

	submission_df = load_submission("files/submission_test04.txt")
	submission_df = submission_df.set_index('DATE', drop=False)

	y_true = submission_df.as_matrix()[:, -1]
	y_pred = np.zeros(len(y_true))

	assignments = learn_structure("files/train_groupedby.csv")
	ass_df = {}
	for ass in assignments:
		local_df = load_training_set('files/assign/train_'+str(ass)+'.csv')
		local_df = local_df.set_index('DATE', drop=False)
		local_df = local_df.sort(['DATE'])
		local_df['time'] = local_df.index.time
		local_df['time_slot'] = local_df.DATE.apply(lambda x: x.time().hour * 2 + (x.time().minute / 30))
		local_df['time_order'] = local_df.DATE.apply(lambda x: x.toordinal())
		local_df['rolling_mean'] = pd.rolling_mean(local_df['CSPL_RECEIVED_CALLS'], window=3, min_periods=1)
		local_df['exp_mean'] = pd.ewma(local_df['CSPL_RECEIVED_CALLS'], span=10)
		local_df['month'] = local_df.index.month
		local_df['weekday'] = pd.DatetimeIndex(local_df['DATE']).weekday
		ass_df[ass] = local_df

	submission_df = submission_df.set_index(['DAY_WE_DS', 'DATE', 'ASS_ASSIGNMENT'], drop=False)

	for i in trange(0, submission_df.shape[0]):
		(day, datetime, assignment) = submission_df.index[i]
		time = datetime.time()

		local_df = ass_df[assignment]
		local_df = local_df[local_df.DAY_WE_DS == day]
		local_df = local_df[local_df.time == time]
		local_df = local_df[local_df.DATE < (datetime - DateOffset(days=3))]
		local_df = local_df.tail(int(n))

		calls = local_df['CSPL_RECEIVED_CALLS'].as_matrix()
		training_set_x = local_df[['time_order']].as_matrix()
		reg = SVR()
		if training_set_x.shape[0] > 0:
			y_pred[i] = reg.fit(training_set_x, calls).predict(datetime.toordinal())
		else:
			y_pred[i] = 0
		
	y_pred_round = [int(round(x)) if x > 0 else 0 for x in y_pred]

	submission_df.prediction = y_pred_round
	submission_df = submission_df.set_index('DATE', drop=False)
	submission_df['DATE'] = submission_df.index.strftime('%Y-%m-%d %H:%M:%S.000')
	submission_df[['DATE', 'ASS_ASSIGNMENT', 'prediction']].to_csv('results/submission_test06.txt', sep='\t', index=False)

	print('MSE round: '),
	print(mean_squared_error(y_true, y_pred_round))


if __name__ == "__main__":
    execute(sys.argv[1])
