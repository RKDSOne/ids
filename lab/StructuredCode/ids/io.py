import json
import numpy as np
import pandas as pd
import os


class DataReader:
    """A cache for reading data"""

    def __init__(s):
        s.conf = json.load(open("conf.json"))

    def read(s, dataset, sep_label=True):
        def filepath(dname):
            if dname.find('-') != -1:
                return '/{0}/{1}.data'.format(dname[:dname.find('-')], dname)
            return '/{0}/{0}.data'.format(dname)

        fpath = s.conf["dpath"] + filepath(dataset)
        has_test = False
        tst_fpath = fpath[:-5] + '-test' + fpath[-5:]
        if os.path.exists(tst_fpath):
            has_test = True

        if sep_label:
            df = pd.read_csv(fpath)
            X = np.array(df.ix[:, :-1]).astype('float64')
            y = np.array(map(lambda o: float(o[1:]), df.ix[:, -1])).astype('float64')
            if has_test:
                df = pd.read_csv(tst_fpath)
                tstX = np.array(df.ix[:, :-1]).astype('float64')
                tsty = np.array(map(lambda o: float(o[1:]), df.ix[:, -1])).astype('float64')
                return has_test, X, y, tstX, tsty
            return has_test, X, y
        else:
            df = pd.read_csv(fpath)
            dat = np.array(df)
            dat[:, -1] = np.array(map(lambda o: float(o[1:]), dat[:, -1]))
            if has_test:
                df = pd.read_csv(tst_fpath)
                tst_dat = np.array(df)
                tst_dat[:, -1] = np.array(map(lambda o: float(o[1:]), tst_dat[:, -1]))
                return has_test, dat.astype('float64'), tst_dat.astype('float64')
            return has_test, dat.astype('float64')
