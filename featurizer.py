import pandas as pd
from tqdm import tqdm, trange


def load_raw_data(filename):
    print("Reading data from file")

    dtype = {
        'DATE': object,
        'WEEK_END': int,
        'DAY_WE_DS': str,
        'ASS_ASSIGNMENT': str,
        'CSPL_RECEIVED_CALLS': int
    }

    cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
    return pd.read_csv(filename, sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'])


def featurize(df):
    features = pd.DataFrame()
    _featurize_day_of_the_week(df, features)
    _featurize_time_slot(df, features)
    _featurize_assignment(df, features)
    _featurize_number_of_calls(df, features)

    return features


def _featurize_day_of_the_week(df, features):
    print("Featurizing days of the week")

    days = [('monday', 'Lundi'), ('tuesday', 'Mardi'), ('wednesday', 'Mercredi'), ('thursday', 'Jeudi'),
            ('friday', 'Vendredi'), ('saturday', 'Samedi'), ('sunday', 'Dimanche')]

    features['is_week_end'] = df.WEEK_END
    for i in trange(7):
        en, fr = days[i]
        features[en] = (df.DAY_WE_DS == fr).astype(int)

    print()


def _featurize_time_slot(df, features):
    print("Featurizing time slots")

    for h in trange(24):
        for s in range(2):
            features['time_slot_' + str(2 * h + s)] = ((df.DATE.dt.hour == h) & (df.DATE.dt.minute == 30 * s)) \
                .astype(int)

    print()


def _featurize_assignment(df, features):
    print("Featurizing assignment")
    assignments = df.ASS_ASSIGNMENT.unique()
    for i in range(len(assignments)):
        features['assignment_' + str(i)] = (df.ASS_ASSIGNMENT == assignments[i]).astype(int)


def _featurize_number_of_calls(df, features):
    features['n_calls'] = df.CSPL_RECEIVED_CALLS
