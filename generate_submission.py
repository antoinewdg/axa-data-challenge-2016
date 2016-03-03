# coding: utf-8

# # Initialization

# In[55]:

import pandas as pd
import sys


def generate_submission(in_filename, out_filename):
    dtype = {
        'DATE': object,
        'ASS_ASSIGNMENT': str,
        'CSPL_RECEIVED_CALLS': int
    }

    cols = ['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']
    print("aaa" + in_filename)
    chunks = pd.read_csv(in_filename, sep=";", usecols=cols, dtype=dtype, parse_dates=['DATE'],
                         chunksize=10 ** 6)

    df = pd.DataFrame()
    for chunk in chunks:
        aux = chunk.groupby(['DATE', 'ASS_ASSIGNMENT'], as_index=False, sort=True)['CSPL_RECEIVED_CALLS'].sum()
        df = pd.concat([df, aux])

    df = df.groupby(['DATE', 'ASS_ASSIGNMENT'], as_index=False, sort=True)['CSPL_RECEIVED_CALLS'].sum()

    df['prediction'] = df['CSPL_RECEIVED_CALLS']

    num_rows = 12408

    df['DATE'] = pd.to_datetime(df['DATE'].astype(str))
    df = df[df['DATE'] > '2012-01-01']
    df = df.sample(n=num_rows)
    df = df.sort_values(by=['DATE'])

    df.to_csv(out_filename, sep='\t', index=False, columns=['DATE', 'ASS_ASSIGNMENT', 'prediction'])


if __name__ == "__main__":
    generate_submission(sys.argv[1], sys.argv[2])
