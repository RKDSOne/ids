import pandas
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# TODO: just choose the best result
# TODO: train parameterized-model for several times and take the average performance
class VisualizeResults(object):
    def __init__(s, res_file):
        res_json = json.load(open(res_file))
        # from this, res is a pandas.DataFrame with columns [algo, param, dataset, conf_mtr]
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
                imf = irec[3]
                iana = s.analyze_fusion(imf)
                x.append(s.param_list[algo].index(float(irec[1])))
                y.append(iana[metric])
            x, y = zip(*sorted(zip(x, y), key=lambda o: o[0]))
            plt.plot(y, label=data)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    conf = json.load(open('conf.json'))
    a = VisualizeResults(os.path.join(
        conf['path'], 'lab/results', 'res2.json'))
    for cnt, i in enumerate(a.algos):
        plt.figure(cnt + 1)
        a.algo_view(i)
