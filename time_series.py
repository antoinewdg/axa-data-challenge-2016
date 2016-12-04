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
import matplotlib.pyplot as plt

dtype = {
    'DATE': object,
    'WEEK_END': int,
    'DAY_WE_DS': str,
    'ASS_ASSIGNMENT': str,
    'CSPL_RECEIVED_CALLS': int
}
cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
df = pd.read_csv('files/train_groupedby.csv', sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'], index_col=['DATE'])

assignments = learn_structure("files/train_groupedby.csv")
assignment = 'Japon'
# df = df[(df.ASS_ASSIGNMENT == assignment)]

day = 'Lundi'
# df = df[(df.DAY_WE_DS == day)

time_slots = np.unique(df.index.time)
time = time_slots[20]
# ts = df[(df.index.time == time)]
for assignment in assignments:
	ax = None
	for time in time_slots[:]:		
		ts = df[(df.ASS_ASSIGNMENT == assignment)]
		# ts = ts[(ts.DAY_WE_DS == day)]
		ts = ts[(ts.index.time == time)]
		# ts = ts['CSPL_RECEIVED_CALLS'].hist()
		if not ts.empty:
			ax = ts.plot(x=ts.index, y='CSPL_RECEIVED_CALLS', style='o', ax=ax, legend=False)
	plt.show()

# days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
# ax = None
# for day in days:
# 	ts = df[(df.DAY_WE_DS == day)]
# 	ts = ts[(ts.index.time == time)]
# 	ax = ts.plot(x=ts.index, y='CSPL_RECEIVED_CALLS', style='o', ax=ax, legend=False)
# 	print ts['CSPL_RECEIVED_CALLS'].mean(), ts['CSPL_RECEIVED_CALLS'].tail(1).mean(), ts['CSPL_RECEIVED_CALLS'].tail(5).mean(), ts['CSPL_RECEIVED_CALLS'].tail(10).mean(), ts['CSPL_RECEIVED_CALLS'].tail(15).mean(), ts['CSPL_RECEIVED_CALLS'].tail(20).mean(), ts['CSPL_RECEIVED_CALLS'].tail(50).mean()
	# ts.plot(x=ts.index, y='CSPL_RECEIVED_CALLS', style='o', legend=False)

# df = df.sort_values()
# ts.plot(x=ts.index, y='CSPL_RECEIVED_CALLS', style='o', legend=False)

# print df

# ts = pd.Series(df['CSPL_RECEIVED_CALLS'].as_matrix())
# print ts

# print converted
# print mean
# for day in ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']:
#     ts = df[(df.DAY_WE_DS == day)]
#     ts = ts[['DATE', 'CSPL_RECEIVED_CALLS']]
#     ax = ts.plot(ax=ax)


# df.to_csv(out_filename, sep='\t', index=False, columns=['DATE', 'ASS_ASSIGNMENT', 'prediction'])
