import pandas as pd
from tqdm import tqdm
import sys, pickle


def load_structure():
    return pickle.load(open("files/learned_structure.p", "rb"))


def learn_structure(in_filename, chunksize=10 ** 6):
    assignments = set()

    dtype = {'ASS_ASSIGNMENT': str}
    cols = ['ASS_ASSIGNMENT']
    chunks = pd.read_csv(in_filename, sep=";", usecols=cols, dtype=dtype, chunksize=chunksize)

    for df in tqdm(chunks):
        assignments.update(df.ASS_ASSIGNMENT.unique())

    structure = {"ASS_ASSIGNMENT": assignments}

    pickle.dump(structure, open("files/learned_structure.p", 'wb'))


if __name__ == "__main__":
    learn_structure(sys.argv[1])
    st = load_structure()
    print("Learned structure:")
    print(st)
