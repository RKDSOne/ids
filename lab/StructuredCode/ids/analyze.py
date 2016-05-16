import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class VisualizeResults(object):

    def __init__(s, res_file):
        res_json = json.load(open(res_file))
        s.res = pd.DataFrame([i.split(',') + [res_json[i]]
                              for i in res_json.keys()])
        s.data_list = np.unique(s.res[2])
        s.algos = np.unique(s.res[0])
        s.param_list = sorted(map(float, np.unique(
            filter(lambda o: o.find(';') == -1, s.res[1]))))

    def query(s, algo):
        multi_params = True
        if s.res[s.res[0] == algo].iloc[0][1].find(';') == -1:
            multi_params = False

        if multi_params:
            for cnt, data in enumerate(s.data_list):
                fig = plt.figure(cnt + 1)
                plt.clf()
                x, y = [], []
                X = np.arange(len(s.param_list))
                Y = X
                X, Y = np.meshgrid(X, Y)
                Z = np.zeros(X.shape)
                tmpdf = s.res[np.logical_and(
                    s.res[0] == algo, s.res[2] == data)]
                for i in range(tmpdf.shape[0]):
                    x.append(map(lambda o: s.param_list.index(
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
        else:
            plt.clf()
            plt.title(algo)
            for data in s.data_list:
                x, y = [], []
                tmpdf = s.res[np.logical_and(
                    s.res[0] == algo, s.res[2] == data)]
                for i in range(tmpdf.shape[0]):
                    x.append(s.param_list.index(float(tmpdf.iloc[i][1])))
                    y.append(tmpdf.iloc[i][3]['f1'])
                x, y = zip(*sorted(zip(x, y), key=lambda o: o[1]))
                plt.plot(y, label=data)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    conf = json.load(open('../conf.json'))
    a = VisualizeResults(os.path.join(conf['path'], 'lab/results', 'res.json'))
    for cnt, i in enumerate(a.algos):
        plt.figure(cnt + 1)
        a.query(i)
