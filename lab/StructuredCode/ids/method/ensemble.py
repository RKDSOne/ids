from scipy.stats import mode
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from base import *


class EasyEnsemble(imalgo):
    """
    Simply split the majority data into buckets each with the same size of minority instances.
    Train one bucket and the entire minority instances for one sub-model, and the final output is simply voted by sub's.
    """

    def __init__(s):
        super(EasyEnsemble, s).__init__()
        s.mdls = []

    # a segment function
    def numPacks(s):
        if s.imr <= 3:
            return 3
        elif s.imr <= 7:
            return int(s.imr) + 1
        else:
            tmp_coef = np.sqrt(7) + 0.0001
            return int(tmp_coef * np.sqrt(s.imr)) + 1

    @imalgo.datazip_decorator
    def fit(s, data):
        s.load_data(data)
        s.identify()
        minoX = s.X[s.y == s.minolab]
        majX = s.X[s.y == s.majlab]
        minoN = minoX.shape[0]
        majN = majX.shape[0]

        bags = [np.random.choice(majN, minoN, replace=False) for i in xrange(s.numPacks())]
        for bag in bags:
            submajX = majX[bag]
            tmp = np.hstack(
                (np.vstack((submajX, minoX)), np.array([s.majlab] * submajX.shape[0] + [s.minolab] * minoN)[:, None]))
            tX = tmp[:, :-1]
            ty = tmp[:, -1]
            mdl = AdaBoostClassifier()
            mdl.fit(tX, ty)
            s.mdls.append(mdl)
        """
        submajN = int(minoN * s.subimba)
        buckets = [majX[i * submajN:min(majN, (i + 1) * submajN)]
                   for i in xrange((majN - 1) / submajN + 1)]
        # The last bucket can be very small, causing imbalance.
        # So, for last packet with
        #   - less than 50% size of minority, it is merged to the before-the-last bucket, while
        #   - reserve this bucket otherwise.
        if buckets[-1].shape[0] < 0.5:
            tmp = buckets.pop(-1)
            buckets[-2] = np.vstack((buckets[-1], tmp))

        for submajX in buckets:
            tmp = np.hstack(
                (np.vstack((submajX, minoX)), np.array([s.majlab] * submajX.shape[0] + [s.minolab] * minoN)[:, None]))
            np.random.shuffle(tmp)
            tX = tmp[:, :-1]
            ty = tmp[:, -1]
            mdl = SVC(**s.mdl_args)
            mdl.fit(tX, ty)
            s.mdls.append(mdl)
        """

    def predict(s, X):
        res = None
        for mdl in s.mdls:
            if res is None:
                res = mdl.predict_proba(X)[:, 1]
            else:
                res = np.vstack((res, mdl.predict_proba(X)[:, 1]))
        return (np.average(res, axis=0) > 0.5).astype('int')


class HKME(imalgo):
    """An intuitive ensemble of SVM classifier with balanced data and OneClassSVM with majority instances."""

    def __init__(s, svc_args={}, svdd_args={}):
        super(HKME, s).__init__()
        s.bsvm = SVC(**svc_args)
        s.vsvm = OneClassSVM(**svdd_args)

    # prepare a balanced version for BSVM
    def balanced(s):
        # random sampling to the same size
        minoN = sum(s.y == s.minolab)
        majN = s.y.shape[0] - minoN
        majX = s.X[s.y == s.majlab]
        # random sample, min(1.5, majcount/minocount) times of minority numbers
        # of, majority instances
        submajX = majX[np.random.choice(
            majN, min(int(1.5 * minoN), majN), replace=False)]
        retX = np.vstack((submajX, s.X[s.y == s.minolab]))
        rety = np.array([s.majlab] * submajX.shape[0] + [s.minolab] * minoN)
        ret = np.hstack((retX, rety[:, None]))
        np.random.shuffle(ret)
        return ret

    @imalgo.datazip_decorator
    def fit(s, data):
        s.load_data(data)
        s.identify()
        majX = s.X[s.y == s.majlab]
        data_balance = s.balanced()
        # call BSVM for results
        s.bsvm.fit(data_balance[:, :-1], data_balance[:, -1])
        # call vSVM for results
        s.vsvm.fit(majX)

    # fea can be row_vector or features-as-columns matrix
    def predict(s, X):
        # AVG fusion?
        res_bsvm = s.bsvm.predict(X)
        res_vsvm = (s.vsvm.predict(X) + 1.0) / 2
        res_avg = (1.0 * res_bsvm + 2.0 * res_vsvm) / 3
        return np.round(res_avg)
