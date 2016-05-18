from sklearn.svm import OneClassSVM
from base import *


class vSVM(imalgo):

    def __init__(s, mdl_args={}):
        super(vSVM, s).__init__()
        s.mdl_args = mdl_args
        s.mdl = None

    @imalgo.datazip_decorator
    def fit(s, data):
        s.load_data(data)
        s.identify()
        majX = s.X[s.y == s.minolab]
        s.mdl = OneClassSVM(**s.mdl_args)
        s.mdl.fit(majX)

    def predict(s, X):
        return (s.mdl.predict(X) + 1.0) / 2


class MTS(imalgo):

    def __init__(s, thre):
        super(MTS, s).__init__()
        s.thre = thre
        s.mdl = None
        s.mu = None
        s.dis_metric = None

    @imalgo.datazip_decorator
    def fit(s, data):
        s.load_data(data)
        s.identify()
        majX = s.X[s.y == s.majlab]
        s.mu = np.mean(majX, axis=0)
        majX -= s.mu
        C = np.dot(majX.T, majX) / s.X.shape[0]
        s.dis_metric = np.linalg.inv(C)

    def predict(s, X):
        reguX = X - s.mu
        if X.ndim == 1:
            return np.dot(np.dot(reguX, s.dis_metric), reguX)
        return (np.sum(np.dot(reguX, s.dis_metric) * reguX[:, None], axis=1) > 1.0 / s.thre).astype('float')
