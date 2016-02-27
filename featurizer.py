import pandas as pd
from tqdm import tqdm, trange
import tables
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import numpy as np

class Featurizer:
    def __init__(self):
        self.assignments = set()

    def featurize(self, in_filename, out_filename, chunksize=10 ** 8):

        self._learn_structure(in_filename, chunksize)
        dtype = {
            'DATE': object,
            'WEEK_END': int,
            'DAY_WE_DS': str,
            'ASS_ASSIGNMENT': str,
            'CSPL_RECEIVED_CALLS': int
        }

        cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
        chunks = pd.read_csv(in_filename, sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'], chunksize=chunksize)

        with tables.open_file(out_filename, mode='w') as out:
            atom = tables.Float64Atom()
            i = 0
            arr = None
            for chunk in chunks:
                features = self._featurize_chunk(chunk)
                if i == 0:
                    arr = out.create_earray(out.root, 'features', atom=atom, shape=(0, features.shape[1]))
                arr.append(features.as_matrix())
                i += 1
            out.close()


    def _learn_structure(self, filename, chunksize=10 ** 6):
        self.assignments = set()

        dtype = {'ASS_ASSIGNMENT': str}
        cols = ['ASS_ASSIGNMENT']
        chunks = pd.read_csv(filename, sep=";", usecols=cols, dtype=dtype, chunksize=chunksize)

        for df in tqdm(chunks):
            self.assignments.update(df.ASS_ASSIGNMENT.unique())

        print(self.assignments)

    def _featurize_chunk(self, df):
        features = pd.DataFrame()
        self._featurize_day_of_the_week(df, features)
        self._featurize_time_slot(df, features)
        self._featurize_assignment(df, features)
        self._featurize_number_of_calls(df, features)

        return features

    def _featurize_day_of_the_week(self, df, features):
        print("Featurizing days of the week")

        days = [('monday', 'Lundi'), ('tuesday', 'Mardi'), ('wednesday', 'Mercredi'), ('thursday', 'Jeudi'),
                ('friday', 'Vendredi'), ('saturday', 'Samedi'), ('sunday', 'Dimanche')]

        features['is_week_end'] = df.WEEK_END
        for i in trange(7):
            en, fr = days[i]
            features[en] = (df.DAY_WE_DS == fr).astype(int)

        print()

    def _featurize_time_slot(self, df, features):
        print("Featurizing time slots")

        for h in trange(24):
            for s in range(2):
                features['time_slot_' + str(2 * h + s)] = ((df.DATE.dt.hour == h) & (df.DATE.dt.minute == 30 * s)) \
                    .astype(int)

        print()

    def _featurize_assignment(self, df, features):
        print("Featurizing assignment")
        i = 0
        for assignment in self.assignments:
            features['assignment_' + str(i)] = (df.ASS_ASSIGNMENT == assignment).astype(int)
            i += 1

    def _featurize_number_of_calls(self, df, features):
        features['n_calls'] = df.CSPL_RECEIVED_CALLS

    def linear_regression(self, in_filename):
    	features = None
    	with tables.open_file(in_filename) as h5_file:
            features = h5_file.root.features

	    X = features[:, :-1]
	    y = features[:, -1]
	    clf = LinearRegression()
	    scores = cross_validation.cross_val_score(clf, X,y, scoring='mean_squared_error', cv=5)

	print(-scores)

feat = Featurizer()
feat.featurize('train_2011_2012.csv', 'featurized.h5')
feat.linear_regression('featurized.h5')

