import sys, os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from method.base import *
from IO import DataReader


class Description(imalgo):
    """Data Description module of Imbalanced Learning Framework"""

    def __init__(s, k_wbnr=8, k_ovlap=1):
        super(Description, s).__init__()
        s.k_wbnr = k_wbnr
        s.k_ovlap = k_ovlap

    def calc_minoNN(s):
        s.mdl_mino = NearestNeighbors(n_neighbors=s.k_wbnr, n_jobs=-1)
        s.mdl_mino.fit(s.X[s.y == s.minolab])

    def small_disjuncts(s):
        # Nearest Neighbors of minorities are calculated only if we need small disjuncts features
        s.calc_minoNN()
        dis, _ = s.mdl.kneighbors()
        dis_mino, _ = s.mdl_mino.kneighbors()
        l2norm = lambda o: np.linalg.norm(o, ord=2, axis=1)
        return np.sqrt(l2norm(dis[s.y==s.minolab]) / l2norm(dis_mino))

    def overlap(s):
        minoN = sum(s.y == s.minolab)
        majN = sum(s.y == s.majlab)
        _, nei_idx = s.mdl.kneighbors(n_neighbors=s.k_ovlap)
        minoNei = nei_idx[s.y == s.minolab]
        return 1 - sum(s.y[minoNei]) * 1.0 / (s.k_ovlap * minoN)

    def fit(s, dname):
        s.dname = dname
        ret = DataReader("../conf.json").read(dname, sep_label=False)
        data = ret[1]
        s.load_data(data)
        s.identify()
        s.mdl = NearestNeighbors(n_neighbors=max(s.k_wbnr, s.k_ovlap), n_jobs=-1)
        s.mdl.fit(s.X)

    def describe(s):
        print s.imr

        print s.overlap()

        WBNR = s.small_disjuncts()
        plt.hist(WBNR, bins=25)
        plt.xlim(0, 1)
        plt.show()
        print os.getcwd()
        plt.savefig('../figures/WBNR/{0}.png'.format(s.dname), transparent=True)


def foo_describe_data(dname):
    print '\n\t' + dname
    disc = Description()
    disc.fit(dname)
    disc.describe()


def main():
    data_list = ['abalone', 'haber', 'iono', 'isolet', 'letter', 'mf-mor', 'mf-zer', 'pima', 'sat', 'uair']
    for dname in data_list:
        foo_describe_data(dname)


if __name__ == '__main__':
    main()
