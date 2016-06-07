import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ids import io as idsdr
from base import *


class Discription(object):
    """Data Description module of Imbalanced Learning Framework"""

    def __init__(s):
        super(Discription, s).__init__()

    def fit(s, data):
        s.load_data(data)
        s.identify()
        s.nnall = NearestNeighbors(n_neighbors=8, n_jobs=-1)
        s.nnall.fit(s.X)

    def describe(s, data):
        print s.imr
        minoN = sum(s.y == s.minolab)
        majN = sum(s.y == s.majlab)
        N = minoN + majN
        minoNei = s.nnall.kneighbors()[s.y == s.minolab][:, 0]
        print sum(minoNei) * 1.0 / N


class NNScope:

    def get_minolab(self):
        tmp = pd.Series(self.y)
        tmp = tmp.value_counts()
        return min(tmp.keys(), key=lambda o: tmp[o])

    def normalization(self):
        self.X -= np.mean(self.X, axis=0)
        self.X /= np.sqrt(np.var(self.X, axis=0))

    def __init__(self, X, y, k):
        self.X = np.array(X, dtype='float64')
        self.normalization()
        self.y = y
        self.minolab = self.get_minolab()
        self.nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        self.nn.fit(self.X)
        self.nn_maj = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        self.nn_maj.fit(self.X[y != self.minolab])
        self.distr = None

    # how many minority samples with given number of minotiry neighbors
    def calc_ratio(self):
        dis_all, _ = self.nn.kneighbors()
        dis_all = dis_all[self.y == self.minolab]
        dis_maj, _ = self.nn_maj.kneighbors(self.X[self.y == self.minolab])
        self.WBNR = np.sqrt(np.mean(dis_all ** 2, axis=1) /
                            np.mean(dis_maj ** 2, axis=1))

    def show_ratio_distr(self):
        plt.hist(self.WBNR, bins=20)


def fake_data():
    x = np.arange(0.0, 5)
    y = np.arange(0.0, 5)
    x, y = np.meshgrid(x, y)
    x = x.reshape((1, -1))[0]
    y = y.reshape((1, -1))[0]
    X = np.array(zip(x, y))
    y = []
    for i in X:
        if i[0] == 2 or i[1] == 2:
            y.append(1)
        else:
            y.append(0)
    return X, y


def main(dname='pima', dump_photo=False):
    print 'Working on *{0}* dataset...'.format(dname)
    rder = idsdr.DataReader()
    X, y = rder.read(dname)
    nn = NNScope(X, y, 8)
    res = nn.calc_ratio()
    nn.show_ratio_distr()
    plt.title(dname)
    plt.ylabel('Prior')
    plt.xlabel('WBNR')
    plt.xlim(0, 1)
    if dump_photo:
        if not os.path.exists('../figures'):
            os.mkdir('../figures')
        if not os.path.exists('../figures/WBNR'):
            os.mkdir('../figures/WBNR')
        plt.savefig('../figures/WBNR/{0}.png'.format(dname), transparent=True)
        plt.clf()
    else:
        plt.show()


def test_main():
    X, y = fake_data()
    nn = NNScope(X, y, 8)
    res = nn.calc_ratio()
    nn.show_ratio_distr()
    plt.show()


if __name__ == '__main__':
    main('abalone', dump_photo=True)
    main('isolet', dump_photo=True)
    main('letter', dump_photo=True)
    main('pima', dump_photo=True)
    main('sat', dump_photo=True)
