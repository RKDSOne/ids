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
    # `data` is a tuple with
    if cache.has_key(dname):
        data = cache[dname]
    else:
        idsdr = DataReader()
        data = idsdr.read(dname)
        cache[dname] = data

    has_test = data[0]

    # `res`: a list of confusion matrix
    res = []

    # idx_set contains (train_index, test_index) tuples with `folds` numbers.
    # So, when the dataset already has test data that don't need cross validation,
    # the idx_set simply has one tuple corrsponding to train and test data stacked in `data`
    idx_set = []
    if has_test:
        trn_size, tst_size = data[1].shape[0], data[3].shape[0]
        data = data[0], np.vstack((data[1], data[3])), np.hstack(data[2], data[4])
        idx_set.append([range(0, trn_size), range(trn_size, trn_size + tst_size)])
    else:
        for tr_idx, tst_idx in KFold(data[1].shape[0], n_folds=folds, shuffle=True):
            idx_set.append([tr_idx, tst_idx])
    for trn_idx, tst_idx in idx_set:
        trnX, trny = data[1][trn_idx], data[2][trn_idx]
        tstX, tsty = data[1][tst_idx], data[2][tst_idx]
        mdl.fit(trnX, trny)
        # build-in sklearn.metrics.confusion_matrix(y_true, y_pred)
        ans = tsty
        pred = mdl.predict(tstX)
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
    data_list = ['abalone', 'isolet', 'letter',
                 'mf-zer', 'mf-mor', 'pima', 'sat']

    for data in data_list[2:3]:
        # ret = evaluate(MWMOTE(7, 5, 5, 3, 5, mdl_args=dict(gamma=32)), data)
        ret = evaluate(SMOTE(5, 3, mdl_args=dict(gamma=32)), data)
        print ret


def test_MTS(dname):
    idsdr = DataReader()
    data = idsdr.read(dname, sep_label=False)
    mdl = MTS(0.08)
    mdl.fit(data)
    res = mdl.predict(data[mdl.y == mdl.majlab, :-1])
    print res
    print '{0} of {1} predicted right'.format(len(res), sum(res))


if __name__ == '__main__':
    unit_test()
    # main()
