import pandas as pd
import numpy as np
import os


def recursive_normalization(path):
    # TODO: I can speed up this in parallelism some day, I think.
    def normalize_csv(csv_name):
        df = pd.read_csv(csv_name)
        data = np.array(df)
        X, y = data[:, :-1], data[:, -1]
        X = X.astype('float64')
        X = np.apply_along_axis(lambda o: (o - np.mean(o)) / np.std(o), 0, X)
        data = np.hstack((X, y[:, None]))
        df.ix[:,:] = data
        df.to_csv(csv_name, index=False)

    _, folders, files = next(os.walk(path))
    # normalize each file here
    for i in files:
        if i[i.rindex('.'):] == '.data':
            normalize_csv(os.path.join(path, i))

    for i in folders:
        recursive_normalization(os.path.join(path, i))
