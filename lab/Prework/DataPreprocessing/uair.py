"""
Dump training and test data from big UAIR csv files.

"""
import numpy as np
import pandas as pd
from datetime import *
import os


def filterBJ(ar):
    # bj_gid = [301322212300, 301322210130, 301322032101, 301322012300, 301322010210, 301322211100, 301322300011,
    #           301322122200, 301322120200, 301322120033, 301322120003, 301322122131, 301322122121, 301322123023,
    #           301322122020, 301322031031, 301322100102, 301322103031, 301322130301, 301322120302, 301322120231,
    #           301322122112, 301322330100, 301322233022, 301322032020, 301323230030, 303100112011, 303101002200,
    #           303100020331, 303100010133, 303100003200, 303101022132, 301323212213, 301322111101, 301320213121,
    #           301320221003]
    bj_gid = [301322212300, 301322210130, 301322032101, 301322012300, 301322010210, 301322211100, 301322300011,
              301322122200, 301322120200, 301322120033, 301322120003, 301322122131, 301322122121, 301322123023,
              301322122020, 301322031031, 301322100102, 301322103031, 301322130301, 301322120302, 301322120231,
              301322122112, 301322330100, 301322233022, 301322032020, 301323230030, 303100112011, 303101002200,
              303100020331, 303100010133, 303100003200, 303101002200, 301323212213, 301322111101, 301320213121,
              301320221003]
    return ar[np.vectorize(lambda o: o in bj_gid)(ar[:, 1]), :]


def propor(dt, qr):
    n = len(dt)
    l = 0
    r = n - 1
    while l <= r:
        mid = (l + r) / 2
        if dt[mid] < qr:
            l = mid + 1
        else:
            r = mid - 1
    return l * 1.0 / n


def sep(fname):
    cols = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'f00', 'f01',
            'f02', 'c0', 'f10', 'f11', 'f12', 'c1', 'f20', 'f21', 'f22', 'c2']

    def regularize(data, mu, sigma):
        X, y = data[:, 3:], data[:, 0]
        X = (X - mu) / sigma
        return np.hstack((X, y[:, None]))

    if os.path.exists('cached-ar.npy'):
        ar = np.load('cached-ar.npy')
    else:
        df = pd.read_csv(fname, header=None)
        ar = np.array(df)
        del(df)
        ar = ar[ar[:, 1] == 301322210130, :]
        np.save('cached-ar', ar)

    X = ar[:, 3:].astype('float')
    mu, sigma = np.mean(X, axis=0), np.std(X, axis=0)

    # index of datetime
    ar[:, 2] = np.vectorize(datetime.strptime)(
        ar[:, 2], '%m/%d/%Y %I:%M:%S %p')
    sepdt = datetime(2015, 1, 15)
    trn = regularize(ar[ar[:, 2] < sepdt, :], mu, sigma)
    tst = regularize(ar[ar[:, 2] >= sepdt, :], mu, sigma)

    pd.DataFrame(trn).to_csv('uair.data', index=False)
    pd.DataFrame(tst).to_csv('uair-test.data', index=False)
