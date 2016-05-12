import json
import numpy as np
import pandas as pd
import os


class DataReader:
    """description of class"""

    def __init__(s):
        s.conf = json.load(open("conf.json"))

    def read(s, dataset, sep_label=True):
        def filepath(dname):
            if dname.find('-') != -1:
                return '/{0}/{1}.data'.format(dname[:dname.find('-')], dname)
            return '/{0}/{0}.data'.format(dname)

        df = pd.read_csv(s.conf["dpath"] + filepath(dataset))
        if sep_label:
            ret = np.array(df.ix[:, :-1]).astype('float64'), np.array(map(lambda o: float(o[1:]), df.ix[:, -1])).astype('float64')
            return ret
        else:
            ret = np.array(df)
            ret[:, -1] = np.array(map(lambda o: float(o[1:]), ret[:, -1]))
            ret=ret.astype('float64')
            return ret

