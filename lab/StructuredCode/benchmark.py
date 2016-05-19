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


def analyze_confusion(res):
    pf = {}
    tmp = None
    for i in res:
        if tmp is None:
            tmp = i
        else:
            tmp = tmp + i
    return tmp.tolist()


def evaluate(mdl, dname, folds=5):
    # `folds` is used for data without train-test separation
    if cache.has_key(dname):
        data = cache[dname]
    else:
        idsdr = DataReader()
        data = idsdr.read(dname)
        cache[dname] = data

    has_test = data[0]

    # `res`: a list of confusion matrix
    res = []
    idx_set = []
    if has_test:
        idx_set =

    else:

        for tr_idx, tst_idx in KFold(data.shape[0], n_folds=folds, shuffle=True):
            idx_set.append([tr_idx, tst_idx])
        # res = Parallel(n_jobs=folds)(
        #     delayed(foo)(mdl, data, idx_set, i) for i in range(folds)[:1])
        for trn_idx, tst_idx in idx_set:
            trn = data[trn_idx]
            tst = data[tst_idx]
            mdl.fit(tr[:, :-1], tr[:, -1])
            # build-in sklearn.metrics.confusion_matrix(y_true, y_pred)
            ans = tst[:, -1]
            pred = mdl.predict(tst[:, :-1])
            res.append(confusion_matrix(ans, pred))

        if hasattr(mdl, 'gamma'):
            param_gamma = mdl.gamma
        elif hasattr(mdl, 'mdl_args'):
            param_gamma = mdl.mdl_args["gamma"]
        else:
            param_gamma = str(mdl.bsvm.gamma) + ';' + str(mdl.vsvm.gamma)
        return [mdl.__class__.__name__ + ',' + str(param_gamma) + ',' + dname, analyze_confusion(res)]


def main():
    paral_jobs = -1
    # thread_method = 'multiprocessing'
    thread_method = 'threading'
    data_list = ['abalone', 'isolet', 'letter',
        'mf-zer', 'mf-mor', 'pima', 'sat']
    gamma_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32]

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

        # ret = Parallel(n_jobs=paral_jobs, backend=thread_method)(
        #     delayed(evaluate)(HKME(svc_args=dict(gamma=gamma_svc), svdd_args=dict(gamma=gamma_svdd)), data) for
        #     gamma_svc in gamma_list for gamma_svdd in gamma_list)
        # for k, v in ret:
        #     all_res[k] = v
        # print 'ok HKME'
        # sys.stdout.flush()

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

    conf = json.load(open('conf.json'))
    folder_path = os.path.join(conf['path'], 'lab/results')
    files_cnt = len(os.listdir(folder_path))
    file_path = os.path.join(
        conf['path'], 'lab/results', 'res{0}.json'.format(files_cnt + 1))
    json.dump(all_res, open(os.path.join(file_path), 'w'))


def unit_test():
    data_list=['abalone', 'isolet', 'letter',
                 'mf-zer', 'mf-mor', 'pima', 'sat']

    for data in data_list[:1]:
        ret=evaluate(MTS(), data)
        print ret


def test_MTS(dname):
    idsdr=DataReader()
    data=idsdr.read(dname, sep_label=False)
    mdl=MTS(0.08)
    mdl.fit(data)
    res=mdl.predict(data[mdl.y == mdl.majlab, :-1])
    print res
    print '{0} of {1} predicted right'.format(len(res), sum(res))


if __name__ == '__main__':
    # test_MTS('isolet')
    main()
