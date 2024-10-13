import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int, num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cnt_resh = num_objects // num_folds * num_folds
    idx = np.arange(0, cnt_resh, dtype=int).reshape(num_folds, -1)
    rem = np.arange(cnt_resh, num_objects, dtype=int)

    split = [(np.concatenate([idx[1:].reshape(1, -1)[0], rem]), idx[0])]

    for i in range(1, num_folds - 1):
        split.append((np.concatenate([idx[:i].reshape(1, -1)[0], idx[i+1:].reshape(1, -1)[0], rem]), idx[i]))

    split.append((idx[:num_folds - 1].reshape(1, -1)[0], np.concatenate([idx[num_folds - 1], rem])))

    return split


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    mean_scores = {}
    names_list = list(parameters.keys())

    def backtrack(params, cur):
        if cur == len(names_list):
            mean = 0

            for f in folds:
                Xs = X
                scaler = params['normalizers'][0]

                if scaler is not None:
                    scaler.fit(X[f[0]])
                    Xs = scaler.transform(X)

                model = knn_class(n_neighbors=params['n_neighbors'], metric=params['metrics'],
                                  weights=params['weights'])
                model.fit(Xs[f[0]], y[f[0]])
                y_pred = model.predict(Xs[f[1]])

                mean += score_function(y[f[1]], y_pred)

            mean /= len(folds)
            mean_scores[(params['normalizers'][1], params['n_neighbors'],
                         params['metrics'], params['weights'])] = mean

            return

        for p in parameters[names_list[cur]]:
            new_params = params
            new_params[names_list[cur]] = p
            backtrack(new_params, cur + 1)

    backtrack({}, 0)

    return mean_scores
