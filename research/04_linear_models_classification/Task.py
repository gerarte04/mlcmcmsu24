import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):
    ft_values_ = {}

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        self.ft_values_ = {}

        for i in X.columns:
            self.ft_values_[i] = np.unique(X[i])

    def transform(self, X):
        new_X = np.array([]).reshape(X.shape[0], 0)

        for i in X.columns:
            for v in self.ft_values_[i]:
                new_X = np.hstack([new_X, np.select([X[i] == v], [1], 0)[:, None]])

        return new_X

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        self.successes_ = {}
        self.counters_ = {}

        for i in X.columns:
            self.successes_[i] = {}
            self.counters_[i] = {}
            uniq = np.unique(X[i])

            for u in uniq:
                idx = np.where(X[i] == u)[0]
                self.successes_[i][u] = np.mean(Y.iloc[idx])
                self.counters_[i][u] = idx.size / X[i].size

    def transform(self, X, a=1e-5, b=1e-5):
        relation = {}

        for c in self.successes_.keys():
            relation[c] = {}

            for k in self.successes_[c].keys():
                relation[c][k] = (self.successes_[c][k] + a) / (self.counters_[c][k] + b)

        new_X = np.array([]).reshape(X.shape[0], 0)

        for i in X.columns:
            new_X = np.hstack([
                new_X,
                np.select([X[i] == k for k in self.successes_[i].keys()], list(self.successes_[i].values()))[:, None],
                np.select([X[i] == k for k in self.counters_[i].keys()], list(self.counters_[i].values()))[:, None],
                np.select([X[i] == k for k in relation[i].keys()], list(relation[i].values()))[:, None]
            ])

        return new_X

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        self.folds_ = list(group_k_fold(X.shape[0], self.n_folds, seed))
        self.successes_ = []
        self.counters_ = []

        for f in self.folds_:
            successes_f = {}
            counters_f = {}

            for i in X.columns:
                successes_f[i] = {}
                counters_f[i] = {}
                uniq = np.unique(X.iloc[f[1]][i])

                for u in uniq:
                    idx = np.where(X.iloc[f[1]][i] == u)[0]
                    successes_f[i][u] = np.mean(Y.iloc[f[1]].iloc[idx])
                    counters_f[i][u] = idx.size / X.iloc[f[1]][i].size

            self.successes_.append(successes_f)
            self.counters_.append(counters_f)

    def transform(self, X, a=1e-5, b=1e-5):
        new_X = np.array([]).reshape(X.shape[0], 0)

        for i in X.columns:
            suc = np.zeros((X.shape[0], 1))
            cnt = np.zeros((X.shape[0], 1))

            for j in range(self.n_folds):
                suc[self.folds_[j][0]] = np.select(
                    [X.iloc[self.folds_[j][0]][i] == k for k in self.successes_[j][i].keys()],
                    list(self.successes_[j][i].values())
                )[:, None]
                cnt[self.folds_[j][0]] = np.select(
                    [X.iloc[self.folds_[j][0]][i] == k for k in self.counters_[j][i].keys()],
                    list(self.counters_[j][i].values())
                )[:, None]

            rel = (suc + a) / (cnt + b)
            new_X = np.hstack([new_X, suc, cnt, rel])

        return new_X

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    uniq, idx = np.unique(x, return_inverse=True)
    w = np.zeros(len(uniq))

    for i in range(len(uniq)):
        class_idx = np.where(idx == i)[0]
        cnt_ones = np.count_nonzero(y[class_idx] == 1)

        w[i] = cnt_ones / class_idx.size if class_idx.size > 0 else 0

    return w
