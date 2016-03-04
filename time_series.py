# coding: utf-8

# # Initialization

# In[55]:

import pandas as pd
import numpy as np
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

assignment = 'CAT'
df = df[(df.ASS_ASSIGNMENT == assignment)]

day = 'Lundi'
# df = df[(df.DAY_WE_DS == day)

time_slots = np.unique(df.index.time)
time = time_slots[7]
# ts = df[(df.index.time == time)]
ax = None
for time in time_slots[:]:
	ts = df[(df.index.time == time)]
	ax = ts.plot(x=ts.index, y='CSPL_RECEIVED_CALLS', style='o', ax=ax, legend=False)
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
plt.show()

# df.to_csv(out_filename, sep='\t', index=False, columns=['DATE', 'ASS_ASSIGNMENT', 'prediction'])
