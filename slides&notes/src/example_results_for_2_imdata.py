from sklearn.cross_validation import KFold


def foo(fname):
    df = pd.read_csv(fname)
    print df['class'].value_counts()
    df = np.array(df)
    np.random.shuffle(df)
    ret = np.zeros((2, 2))

    kf = KFold(df.shape[0], n_folds=5)
    for trnidx, tstidx in kf:
        X, y = df[trnidx, :-1], df[trnidx, -1]
        mdl = SVC()
        mdl.fit(X, y)
        pred = mdl.predict(df[tstidx, :-1])
        ret += confusion_matrix(df[tstidx, -1], pred)
    print ret.astype(int)
    p = ret[1][1] / (ret[0][1] + ret[1][1])
    r = ret[1][1] / (ret[1][0] + ret[1][1])
    print 2 * p * r / (p + r)