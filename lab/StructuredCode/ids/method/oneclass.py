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
