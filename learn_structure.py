import pandas as pd
from tqdm import tqdm
import sys, pickle


def load_structure(filename="files/learned_structure.p"):
    return pickle.load(open(filename, "rb"))


def learn_structure(in_filename, out_filename, chunksize=10 ** 6):
    assignments = set()

    dtype = {'ASS_ASSIGNMENT': str}
    cols = ['ASS_ASSIGNMENT']
    chunks = pd.read_csv(in_filename, sep=";", usecols=cols, dtype=dtype, chunksize=chunksize)

    for df in tqdm(chunks):
        assignments.update(df.ASS_ASSIGNMENT.unique())

    pickle.dump(assignments, open(out_filename, 'wb'))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        sys.argv.append("files/learned_structure.p")
    learn_structure(sys.argv[1], sys.argv[2])
    st = load_structure(sys.argv[2])
    print("Learned structure:")
    print(st)
