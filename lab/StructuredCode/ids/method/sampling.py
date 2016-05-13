from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from base import *
import time as pytime


class SMOTE(imalgo):
    def __init__(s, k, rate, mdl_args={}):
        super(SMOTE, s).__init__()
        s.k = k
        s.rate = rate
        s.mdl_args = mdl_args
        s.sampled_mdl = None

    def sample(s):
        if s.data is None:
            raise ValueError('data not loaded.')
        mdl = NearestNeighbors(n_neighbors=s.k, n_jobs=-1)
        minoX = s.X[s.y == s.minolab]
        majX = s.X[s.y == s.majlab]
        mdl.fit(minoX)
        _, nei_table = mdl.kneighbors()

        generated = None
        for cnt, nei_idx in enumerate(nei_table):
            x = minoX[cnt]
            if s.rate >= 0.5 * s.k:
                nei = minoX[np.random.choice(nei_idx, int(s.rate))]
                new = x + np.random.rand(int(s.rate), 1) * (nei - x)

            else:
                nei = minoX[nei_idx]
                new = x + np.random.rand(s.k, 1) * (nei - x)
                # each of the synthesed k points has N/k * 100 % probability to be chosen
                new = new[np.random.rand(s.k) > s.rate * 1.0 / s.k]
            if generated is None:
                generated = new
            else:
                generated = np.vstack((generated, new))
        # number of generated instances
        N = len(generated)
        ret = np.hstack((np.vstack((minoX, generated, majX)),
                         np.array([s.minolab] * (minoX.shape[0] + N) + [s.majlab] * majX.shape[0])[:, None]))
        np.random.shuffle(ret)
        return ret

    @imalgo.datazip_decorator
    def fit(s, data):
        s.load_data(data)
        s.identify()
        s.sampled_mdl = SVC(**s.mdl_args)
        sampled_data = s.sample()
        s.sampled_mdl.fit(sampled_data[:, :-1], sampled_data[:, -1])

    def predict(s, X):
        return s.sampled_mdl.predict(X)


class MWMOTE(imalgo):
    def __init__(s, k1, k2, k3, rate, Cfth, mdl_args={}):
        super(MWMOTE, s).__init__()
        s.k1 = k1
        s.k2 = k2
        s.k3 = k3
        s.rate = rate
        s.Cfth = Cfth
        s.mdl_args = mdl_args
        s.sampled_mdl = None
        s.N = -1

    def sample(s):
        if s.data is None:
            raise ValueError('data not loaded.')
        mdl = NearestNeighbors(n_neighbors=s.k1, n_jobs=-1)
        mdl.fit(s.X)
        _, nei_table = mdl.kneighbors()
        # the index of those minority points with minority neighbors
        noise_mino_idx = filter(lambda o: sum(s.y[nei_table[o]] == s.minolab) != 0 and s.y[o] == s.minolab,
                                range(s.X.shape[0]))
        minoX = s.X[s.y == s.minolab]
        majX = s.X[s.y == s.majlab]

        mdl_maj = NearestNeighbors(n_neighbors=s.k2, n_jobs=-1)
        mdl_maj.fit(majX)
        # all majority examples on the bound
        _, tmp = mdl_maj.kneighbors(s.X[noise_mino_idx])
        # remove dumplicate examples
        bound_maj_idx = np.unique(np.reshape(tmp, (1, -1))[0])

        mdl_mino = NearestNeighbors(n_neighbors=s.k3, n_jobs=-1)
        mdl_mino.fit(minoX)
        # find minority examples on the bound backward
        _, tmp = mdl_mino.kneighbors(majX[bound_maj_idx])
        bound_mino_idx = np.unique(np.reshape(tmp, (1, -1))[0])

        bound_maj = majX[bound_maj_idx]
        bound_mino = minoX[bound_mino_idx]

        # difference matrix, shape = (majN, minoN).
        # Due to broadcast(strech), diff[i][j][k] would be maj[i][k]-mino[j][k],
        # thus vector diff[i][j]=maj[i]-mino[j] representing the outer vector diff.
        diff = bound_maj[:, None, :] - bound_mino
        Cf = lambda o: min(s.X.shape[1] / np.linalg.norm(o, 2), s.Cfth) * 1.0 / s.Cfth
        CM = np.apply_along_axis(Cf, 2, diff)

        W = np.mean(((CM * CM).T / np.sum(CM, axis=1)).T, axis=0)

        # P is the normalized Weight Vector, standing for the probability chosen to synthese
        P = W / np.sum(W)

        # np.save(open('W-{0}.ndarray'.format(s.mdl_args["gamma"]), 'w'), CM)

        # choose N bound minority examples to synthese, selection probability accroding to their weight
        chosen = np.random.choice(range(len(P)), size=s.N, p=P)
        chosenp = bound_mino[chosen]

        # would not implement CLUSTERING in MWMOTE, I could see no effort of that but time-consumption.
        _, nei = mdl_mino.kneighbors(chosenp, s.k1)
        dualp = minoX[[i[int(np.random.rand() * s.k1)] for i in nei]]

        generated = chosenp + np.random.rand(s.N, 1) * (dualp - chosenp)
        ret = np.hstack((np.vstack((minoX, generated, majX)),
                         np.array([s.minolab] * (minoX.shape[0] + s.N) + [s.majlab] * majX.shape[0])[:, None]))
        np.random.shuffle(ret)
        return ret

    @imalgo.datazip_decorator
    def fit(s, data):
        def toc(prev_time):
            cur = pytime.clock()
            print cur - prev_time
            return cur

        # tmp_time = pytime.clock()
        s.load_data(data)
        # tmp_time = toc(tmp_time)

        s.identify()
        # tmp_time = toc(tmp_time)

        s.sampled_mdl = SVC(**s.mdl_args)
        s.N = int(s.rate * sum(s.y == s.minolab))
        sampled_data = s.sample()
        # tmp_time = toc(tmp_time)

        s.sampled_mdl.fit(sampled_data[:, :-1], sampled_data[:, -1])
        # tmp_time = toc(tmp_time)

    def predict(s, X):
        return s.sampled_mdl.predict(X)
