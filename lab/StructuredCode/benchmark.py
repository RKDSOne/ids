from ids.method.ensemble import *
from ids.method.oneclass import *
from ids.method.sampling import *
from ids.io import DataReader
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
import json
import sys

cache = {}


def analyse_res(res):
    pf = {}
    tmp = None
    for i in res:
        if tmp is None:
            tmp = i
        else:
            tmp = tmp + i
    try:
        pf['precision'] = 1.0 * tmp[1][1] / (tmp[1][1] + tmp[0][1])
    except ZeroDivisionError:
        pf['precision'] = -1
    pf['recall'] = 1.0 * tmp[1][1] / (tmp[1][1] + tmp[1][0])
    pf['f1'] = 1.0 * tmp[1][1] / (tmp[1][1] * 2 + tmp[1][0] + tmp[0][1])
    return pf


def foo(mdl, data, idx_set, i):
    tr = data[idx_set[i][0]]
    tst = data[idx_set[i][1]]
    mdl.fit(tr[:, :-1], tr[:, -1])
    # build-in sklearn.metrics.confusion_matrix(y_true, y_pred)
    ans = tst[:, -1]
    pred = mdl.predict(tst[:, :-1])
    return confusion_matrix(ans, pred)


def evaluate(mdl, dname, folds=5):
    if cache.has_key(dname):
        data = cache[dname]
    else:
        idsdr = DataReader()
        data = idsdr.read(dname, sep_label=False)
        cache[dname] = data
    res = []
    idx_set = []
    for tr_idx, tst_idx in KFold(data.shape[0], n_folds=folds, shuffle=True):
        idx_set.append([tr_idx, tst_idx])
    res = Parallel(n_jobs=folds)(
        delayed(foo)(mdl, data, idx_set, i) for i in range(folds)[:1])
    if hasattr(mdl, 'gamma'):
        param_gamma = mdl.gamma
    elif hasattr(mdl, 'mdl_args'):
        param_gamma = mdl.mdl_args["gamma"]
    else:
        param_gamma = str(mdl.bsvm.gamma) + ';' + str(mdl.vsvm.gamma)
    return [mdl.__class__.__name__ + ',' + str(param_gamma) + ',' + dname, analyse_res(res)]


def main():
    paral_jobs = -1
    # thread_method = 'multiprocessing'
    thread_method = 'threading'
    data_list = ['abalone', 'isolet', 'letter', 'mf-zer', 'mf-mor', 'pima', 'sat']
    gamma_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    all_res = {}
    for data in data_list:
        # parallel experiments.

        ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
            delayed(evaluate)(EasyEnsemble(subimba=1, mdl_args=dict(gamma=gamma)), data) for gamma in gamma_list)
        for k, v in ret:
            all_res[k] = v
        print 'ok EasyEnsemble'
        sys.stdout.flush()

        ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
            delayed(evaluate)(SVC(gamma=gamma), data) for gamma in gamma_list)
        for k, v in ret:
            all_res[k] = v
        print 'ok SVC'
        sys.stdout.flush()

        ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
            delayed(evaluate)(HKME(svc_args=dict(gamma=gamma_svc), svdd_args=dict(gamma=gamma_svdd)), data) for
            gamma_svc in gamma_list for gamma_svdd in gamma_list)
        for k, v in ret:
            all_res[k] = v
        print 'ok HKME'
        sys.stdout.flush()

        ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
            delayed(evaluate)(vSVM(mdl_args=dict(gamma=gamma)), data) for gamma in gamma_list)
        for k, v in ret:
            all_res[k] = v
        print 'ok vSVM'
        sys.stdout.flush()

        ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
            delayed(evaluate)(SMOTE(5, 3, mdl_args=dict(gamma=gamma)), data) for gamma in gamma_list)
        for k, v in ret:
            all_res[k] = v
        print 'ok SMOTE'
        sys.stdout.flush()

        ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
            delayed(evaluate)(MWMOTE(7, 5, 5, 3, 5, mdl_args=dict(gamma=gamma)), data) for gamma in gamma_list)
        for k, v in ret:
            all_res[k] = v
        print 'ok MWMOTE'
        sys.stdout.flush()


def unit_test():
    data_list = ['abalone', 'isolet', 'letter',
                 'mf-zer', 'mf-mor', 'pima', 'sat']

    for data in ['letter']:
        ret = evaluate(MWMOTE(7, 5, 5, 3, 5, mdl_args=dict(gamma=1)), data)
        print ret


if __name__ == '__main__':
    unit_test()
    # main()
    # json.dump(global_res_table, open('res.json', 'w'))
