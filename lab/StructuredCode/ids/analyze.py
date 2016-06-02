import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from method.base import *
from IO import DataReader


class Description(imalgo):
    """Data Description module of Imbalanced Learning Framework"""

    def __init__(s):
        super(Description, s).__init__()

    @imalgo.datazip_decorator
    def fit(s, data):
        s.load_data(data)
        s.identify()
        s.nnall = NearestNeighbors(n_neighbors=8, n_jobs=-1)
        s.nnall.fit(s.X)

    def describe(s):
        print s.imr
        minoN = sum(s.y == s.minolab)
        majN = sum(s.y == s.majlab)
        minoNei = s.nnall.kneighbors()[1][s.y == s.minolab]
        print 1 - sum(s.y[minoNei[:, 0]]) * 1.0 / minoN


def foo_describe_data(dname):
    print '\n\t'+dname
    ret = DataReader("../conf.json").read(dname, sep_label=False)
    data=ret[1]
    disc = Description()
    disc.fit(data)
    disc.describe()


def main():
    data_list = ['uair', 'abalone', 'isolet', 'letter',
                 'mf-zer', 'mf-mor', 'pima', 'sat']
    for dname in data_list:
        foo_describe_data(dname)


if __name__ == '__main__':
    main()
