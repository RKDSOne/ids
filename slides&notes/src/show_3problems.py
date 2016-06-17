import numpy as np
import pandas as pd
import time as pytime
from datetime import *
import os
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


plt.ion()
plt.style.use('default')
%gui qt


def foo(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1])

plt.clf()
plt.grid('off')

# Extremly imbalance


def eim():
    x = np.random.rand(300)
    y = np.random.rand(300)
    plt.plot(x, y, 'o', ms=20)
    x = np.random.randn(100)
    y = np.random.randn(100)
    x, y = zip(*filter(lambda o: foo(o) > 2, zip(x, y)))
    plt.plot(x, y, '*', ms=33)
    plt.title('Extremly Imbalance')
    plt.savefig('extr', transparent=True)


# Overlap between classes
def ovlap(jt=0.05):
    plt.clf()
    majN = 75
    minoN = 20
    x = np.random.rand(majN) * 2 - 1 + np.random.randn(majN) * jt
    y = np.random.rand(majN) * 2 - 1 + np.random.randn(majN) * jt
    plt.plot(x, y, 'o', ms=20)

    x = np.random.rand(minoN) + 0.3 + np.random.randn(minoN) * jt
    y = np.random.rand(minoN) + 0.3 + np.random.randn(minoN) * jt
    plt.plot(x, y, '*', ms=20)
    plt.title('Overlap Between Classes')
    plt.savefig('ovlap', transparent=True)


# Small disjuncts
def disj(jt=0.05):
    data = []

    plt.clf()
    majN = 75
    minoN = 20
    x = np.random.rand(majN) * 2 - 1 + np.random.randn(majN) * jt
    y = np.random.rand(majN) * 2 - 1 + np.random.randn(majN) * jt
    plt.plot(x, y, 'o', ms=20)

    x, y = [], []
    while True:
        s = raw_input()
        if s[:5] == '-9999':
            break
        tx, ty = map(float, s.split())
        x.append(tx)
        y.append(ty)

    x = np.random.rand(minoN) + 0.3 + np.random.randn(minoN) * jt
    y = np.random.rand(minoN) + 0.3 + np.random.randn(minoN) * jt
    plt.plot(x, y, '*', ms=20)
    plt.title('Small Disjunts')
    plt.savefig('disj', transparent=True)


def sampling_overfitting(rate=3):
    data = []

    plt.figure(1)
    plt.clf()
    majN = 100
    minoN = 20
    jt = 1
    x = np.random.rand(majN) * 2 - 1 + np.random.randn(majN) * jt
    y = np.random.rand(majN) * 2 - 1 + np.random.randn(majN) * jt
    plt.plot(x, y, 'o', ms=10)
    for i in xrange(majN):
        data.append([x[i], y[i], 0])

    x = np.random.rand(minoN) + 0.1 + np.random.randn(minoN) * jt
    y = np.random.rand(minoN) + 0.1 + np.random.randn(minoN) * jt
    plt.plot(x, y, '*', ms=10)
    for i in xrange(minoN):
        for j in xrange(rate):
            data.append([x[i], y[i], 1])
    xlim, ylim = plt.xlim(), plt.ylim()

    mdl = DecisionTreeClassifier(criterion='entropy')
    data = np.array(data)
    mdl.fit(data[:, :-1], data[:, -1])

    x = np.linspace(xlim[0], xlim[1], 300)
    y = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    Z = mdl.predict(grid_points)
    Z = Z.reshape((len(x), -1))
    plt.contourf(x, y, Z, 1)
    plt.show()
