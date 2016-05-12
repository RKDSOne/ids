import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def lcsv(csv_name, sep=False):
    data = np.array(pd.read_csv(csv_name))
    if sep:
        return data[:, :-1].astype('float64'), data[:, -1]
    return data


def dcsv(X, y, csv_name):
    df = pd.DataFrame(np.hstack((X, y[:, None])))
    df.columns = ['f{0}'.format(i) for i in xrange(X.shape[1])] + ['class']
    df.to_csv(csv_name, index=False)


def plot_accumulation(x, energy=90):
    y = [0]
    for i in x:
        y.append(y[-1] + i)
    y /= y[-1] / 100.0

    l, r = 0, len(y) - 1
    while l < r:
        mid = (l + r) / 2
        if y[mid] < energy:
            l = mid + 1
        else:
            r = mid - 1
    plt.plot(y)
    plt.text(l, 50, str(l))
    plt.axvline(l, ymax=0.9)
    plt.show()


def visualized_analyze(fname):
    data = lcsv(fname)
    X, y = data[:, :-1], data[:, -1]
    meanX = np.mean(X, axis=0)
    X -= meanX
    C = np.dot(X.T, X)
    LMD, V = np.linalg.eig(C)
    plot_accumulation(LMD)


def gopca(fname, tar_dims):
    # this tool generates `.htfpca` files, which are numpy.ndarrays, recording
    # mean and informative basis.
    def pca(X):
        X = X.astype('float64')
        meanX = np.mean(X, axis=0)
        X -= meanX
        C = np.dot(X.T, X)
        LMD, V = np.linalg.eig(C)
        return meanX, V

    # check if there's test data
    dot_pos = fname.rindex('.')
    tst_name = fname[:dot_pos] + '-test' + fname[dot_pos:]
    has_tst = os.path.exists(tst_name)

    data = lcsv(fname)
    if has_tst:
        tst_data = lcsv(tst_name)

    mean, basis = pca(np.vstack((data, tst_data))[:, :-1])
    # record mean and basis
    np.save(os.path.join(os.path.dirname(fname), 'mean.htfpca'), mean)
    np.save(os.path.join(os.path.dirname(fname), 'basis.htfpca'), basis)

    # simply take first `tar_dims` slice,
    # as it's already sorted by squared deviation
    # NOTE THAT eigen vectors are stacked horizontally
    V = basis[:, :tar_dims]

    dcsv(np.dot(data[:, :-1], V), data[:, -1], fname)
    if has_tst:
        dcsv(np.dot(tst_data[:, :-1], V), tst_data[:, -1], tst_name)
