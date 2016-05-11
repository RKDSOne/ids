import numpy as np
import DataReader as idsdr
import matplotlib.pyplot as plt


def plot_accumulation(x):
    y = [0]
    for i in x:
        y.append(y[-1]+i)
    y /= y[-1] / 100.0

    l, r = 0, len(y) - 1
    while l < r:
        mid = (l + r) / 2
        if y[mid] < 90:
            l = mid + 1
        else:
            r = mid - 1
    plt.plot(y)
    plt.text(l, 50, str(l))
    plt.axvline(l, ymax=0.9)
    plt.show()


rder = idsdr.DataReader()
X, y = rder.read('isolet')
meanX = np.mean(X, axis=0)
X -= meanX
C = np.dot(X.T, X)
LMD, V = np.linalg.eig(C)
plot_accumulation(LMD)
