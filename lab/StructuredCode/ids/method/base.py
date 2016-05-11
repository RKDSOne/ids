from ids.io import *


class imalgo(object):
    """
    `imalgo` is the base class for Imbalanced Learning Algorithms, integrated with `load_data` and `identify`.
    """

    @staticmethod
    def datazip_decorator(func):
        def wrapper(s, *data):
            if len(data) == 1:
                data = data[0]
            else:
                subdata = [i[:, None] if i.ndim == 1 else i for i in data]
                data = np.hstack(subdata)
            func(s, data)

        return wrapper

    def __init__(s):
        s.minolab = None
        s.majlab = None
        s.imr = 0
        s.data = None
        s.X = None
        s.y = None

    def load_data(s, data):
        """
        implementation for loading data given either
          - a numpy.ndarray, or
          - a string corresponding to a local data set, or
          - a dict with "X" and "y" as keys, and numpy.ndarrays as values.
        the local data set named "data_name" can be found in the path: "/conf[u'dpath']/data_name/data_name.data".
        """
        if isinstance(data, str):
            idsdr = DataReader()
            s.X, s.y = idsdr.read(data)
            s.data = np.hstack((s.X, s.y))
        else:
            s.data = data
            s.X, s.y = np.array(data[:, :-1]), np.array(data[:, -1])

    # to help get minority label and majority label. Also the imbalanced ratio is calculated in `s.imr`.
    def identify(s):
        labs, counts = np.unique(s.data[:, -1], return_counts=True)
        if len(labs) != 2:
            raise TypeError('must be 2-classes classification task')
        s.minolab, s.majlab = labs if counts[0] < counts[1] else labs[::-1]
        s.imr = counts[0] * 1.0 / counts[1]
        if s.imr >= 1:
            s.imr = 1 / s.imr
