import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import mean_squared_error
from featurizer import *
from pandas.tseries.offsets import *
import math
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

def execute(n):
	training_df = load_training_set("files/train_groupedby.csv")
	training_df = training_df.set_index('DATE', drop=False)

	submission_df = load_submission("files/submission.txt")
	submission_df = submission_df.set_index('DATE', drop=False)

	y_true = submission_df.as_matrix()[:, -1]
	y_pred = np.zeros(len(y_true))

	assignments = learn_structure("files/train_groupedby.csv")

	ass_df = {}
	for ass in assignments:
		local_df = load_training_set('files/assign/train_'+str(ass)+'.csv')
		local_df = local_df.set_index('DATE', drop=False)
		local_df = local_df.sort_values(by=['DATE'])
		local_df['time'] = local_df.index.time
		local_df['time_slot'] = local_df.DATE.apply(lambda x: x.time().hour * 2 + (x.time().minute / 30))
		local_df['rolling_mean'] = pd.rolling_mean(local_df['CSPL_RECEIVED_CALLS'], window=3, min_periods=1)
		local_df['exp_mean'] = pd.ewma(local_df['CSPL_RECEIVED_CALLS'], span=10)
		local_df['month'] = local_df.index.month
		local_df['weekday'] = pd.DatetimeIndex(local_df['DATE']).weekday
		ass_df[ass] = local_df

	submission_df['time_slot'] = submission_df.DATE.apply(lambda x: x.time().hour * 2 + (x.time().minute / 30))
	submission_df['month'] = submission_df.index.month
	submission_df['weekday'] = pd.DatetimeIndex(submission_df['DATE']).weekday
	submission_df2 = submission_df.set_index(['DATE', 'weekday', 'time_slot', 'month', 'ASS_ASSIGNMENT'], drop=False)

	submission_df = submission_df.set_index(['DAY_WE_DS', 'DATE', 'ASS_ASSIGNMENT'], drop=False)
	for i in trange(0, submission_df.shape[0]):
		(datetime, weekday, time_slot, month, assignment) = submission_df2.index[i]
		(day, datetime, assignment) = submission_df.index[i]
		time = datetime.time()

		local_df = ass_df[assignment]
		local_df = local_df[(local_df.weekday == weekday)]
		local_df = local_df[local_df.time == time]

		local_df = local_df[local_df.DATE < (datetime - DateOffset(days=3))]
		local_df = local_df.tail(int(n))

		calls = local_df['CSPL_RECEIVED_CALLS'].as_matrix()
		training_set_x = local_df[['time_slot', 'month', 'weekday']].as_matrix()
		
		reg = SVR(kernel='rbf')
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(training_set_x[:,0], training_set_x[:,2], calls)
		ax.plot(training_set_x[:,0], training_set_x[:,2], reg.fit(training_set_x, calls).predict(training_set_x))
		plt.show()

		if training_set_x.shape[0] > 0:
			y_pred[i] = reg.fit(training_set_x, calls).predict(np.array((time_slot, month, weekday)))
		else:
			y_pred[i] = 0	

	y_pred_round = [int(round(x)) if x > 0 else 0 for x in y_pred]

	submission_df.prediction = y_pred_round
	submission_df = submission_df.set_index('DATE', drop=False)
	submission_df['DATE'] = submission_df.index.strftime('%Y-%m-%d %H:%M:%S.000')

	submission_df[['DATE', 'ASS_ASSIGNMENT', 'prediction']].to_csv('results/submission13.txt', sep='\t', index=False)

	print('MSE round: '),
	print(mean_squared_error(y_true, y_pred_round))

	print('MSE not round: '),
	print(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    execute(sys.argv[1])