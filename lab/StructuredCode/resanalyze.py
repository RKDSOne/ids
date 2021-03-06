import pandas
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import re
from ids import imfeature


# TODO: just choose the best result
# TODO: train parameterized-model for several times and take the average
class PlotView(object):
    def __init__(s, res_file):
        res_json = json.load(open(res_file))
        # from this, res is a pandas.DataFrame with columns [algo, param,
        # dataset, conf_mtr]
        s.res = pandas.DataFrame([i.split(',') + [res_json[i]]
                                  for i in res_json.keys()])
        s.data_list = np.unique(s.res[2])
        s.algos = np.unique(s.res[0])
        s.param_list = {}
        # TODO: the param_list should not be a global variable, tobefixed.
        for algo in s.algos:
            s.param_list[algo] = sorted(
                map(float, np.unique(filter(lambda o: o.find(';') == -1, s.res[s.res[0] == algo][1]))))

    def analyze_fusion(s, fm):
        ret = {}
        try:
            ret['precision'] = 1.0 * fm[1][1] / (fm[1][1] + fm[0][1])
        except ZeroDivisionError:
            ret['precision'] = -1
        ret['recall'] = 1.0 * fm[1][1] / (fm[1][1] + fm[1][0])
        ret['f1'] = 2.0 * fm[1][1] / (fm[1][1] * 2 + fm[1][0] + fm[0][1])
        ret['FP'] = 1.0 * fm[0][1] / (fm[0][0] + fm[0][1])
        ret['TP'] = 1.0 * fm[1][1] / (fm[1][1] + fm[1][0])
        return ret

    def algo_view_3d(s, algo):
        for data in s.data_list:
            fig = plt.figure(data)
            plt.clf()
            x, y = [], []
            X = np.arange(len(s.param_list[algo]))
            Y = X
            X, Y = np.meshgrid(X, Y)
            Z = np.zeros(X.shape)
            tmpdf = s.res[np.logical_and(
                s.res[0] == algo, s.res[2] == data)]
            for i in range(tmpdf.shape[0]):
                x.append(map(lambda o: s.param_list[algo].index(
                    float(o)), tmpdf.iloc[i][1].split(';')))
                y.append(tmpdf.iloc[i][3]['f1'])
                Z[x[-1][0]][x[-1][1]] = y[-1]
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('f1')
        plt.show()

    def algo_view(s, algo, metric='f1'):
        plt.clf()
        plt.title(algo)
        for data in s.data_list:
            x, y = [], []
            tmpdf = s.res[np.logical_and(
                s.res[0] == algo, s.res[2] == data)]
            for i in range(tmpdf.shape[0]):
                irec = tmpdf.iloc[i]
                ifm = irec[3]
                iana = s.analyze_fusion(ifm)
                x.append(s.param_list[algo].index(float(irec[1])))
                y.append(iana[metric])
            if x[0]!=-1:
                x, y = zip(*sorted(zip(x, y), key=lambda o: o[0]))
            plt.plot(y, label=data)
        plt.legend()
        plt.show()


class TableView(object):
    def __init__(s, res_file_list):
        s.res = []
        for res_file in res_file_list:
            res_json = json.load(open(res_file))
            # from this, res is a pandas.DataFrame with columns [algo, param,
            # dataset, conf_mtr]
            s.res.append(pandas.DataFrame(
                [i.split(',') + [res_json[i]] for i in res_json.keys()]))
        s.data_list = np.unique(s.res[0][2])
        s.algos = np.unique(s.res[0][0])
        # sort them in alphabet order
        s.data_list = sorted(s.data_list)
        s.algos = sorted(s.algos)

    def analyze_fusion(s, fm):
        ret = {}
        try:
            ret['precision'] = 1.0 * fm[1][1] / (fm[1][1] + fm[0][1])
        except ZeroDivisionError:
            ret['precision'] = -1
        ret['recall'] = 1.0 * fm[1][1] / (fm[1][1] + fm[1][0])
        ret['f1'] = 2.0 * fm[1][1] / (fm[1][1] * 2 + fm[1][0] + fm[0][1])
        ret['FP'] = 1.0 * fm[0][1] / (fm[0][0] + fm[0][1])
        ret['TP'] = 1.0 * fm[1][1] / (fm[1][1] + fm[1][0])
        return ret

    def show(s):
        def foo(fm):
            return s.analyze_fusion(fm)['f1']

        for k in xrange(len(s.res)):
            s.res[k][3] = pandas.Series([foo(i) for i in s.res[k][3]])
        tabel_res = np.zeros((len(s.res), len(s.data_list), len(s.algos)))
        for k in xrange(len(s.res)):
            for i, data in enumerate(s.data_list):
                for j, algo in enumerate(s.algos):
                    tabel_res[k][i][j] = max(
                        s.res[k][(s.res[k][0] == algo) & (s.res[k][2] == data)][3])
        tabel_res = np.mean(tabel_res, axis=0)
        tabel_res = pandas.DataFrame(tabel_res)
        tabel_res.columns = s.algos
        tabel_res['Dataset'] = s.data_list
        cols = tabel_res.columns
        tabel_res = tabel_res[['Dataset'] + s.algos]
        print tabel_res
        return tabel_res


def main():
    if len(sys.argv) == 3:
        st, ed = int(sys.argv[1]), int(sys.argv[2])
    else:
        st, ed = 8, 13
    conf = json.load(open('conf.json'))
    file_path_constructor = lambda o: os.path.join(conf['path'], 'lab/results', 'res{0}.json'.format(o))
    tv = TableView([file_path_constructor(i) for i in xrange(st, ed)])
    tabel = tv.show()

    imr_list = []
    ovlap_list = []
    for dname in tv.data_list:
        imr, ovlap = imfeature.foo_describe_data(dname, 'conf.json')
        imr_list.append(imr)
        ovlap_list.append(ovlap)
    tabel['imr'] = pandas.Series(imr_list)
    tabel['ovlap'] = pandas.Series(ovlap_list)
    tabel.to_csv(os.path.join(
        conf['path'], 'lab/results', 'pre.csv'), index=False)


def see_vsvm():
    conf = json.load(open('conf.json'))
    file_path_constructor = lambda o: os.path.join(conf['path'], 'lab/results', 'res{0}.json'.format(o))
    tmp = PlotView(file_path_constructor(8))
    tmp.algo_view('vSVM')


if __name__ == '__main__':
    main()
