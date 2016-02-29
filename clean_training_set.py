import pandas as pd
import sys


def clean_training_set(in_filename, out_filename):
    dtype = {
        'DATE': object,
        'WEEK_END': int,
        'DAY_WE_DS': str,
        'ASS_ASSIGNMENT': str,
        'CSPL_RECEIVED_CALLS': int
    }

    cols = ['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
    chunks = pd.read_csv(in_filename, sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'],
                         chunksize=10 ** 6)

    df = pd.DataFrame()
    for chunk in chunks:
        aux = chunk.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)[
            'CSPL_RECEIVED_CALLS'].sum()
        df = pd.concat([df, aux])

    df = df.groupby(['DATE', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT'], as_index=False, sort=False)[
        'CSPL_RECEIVED_CALLS'].sum()
    df = df.sort_values(by=['DATE'])

    df.to_csv(out_filename, sep=";", index=False)


if __name__ == "__main__":
    clean_training_set(sys.argv[1], sys.argv[2])
