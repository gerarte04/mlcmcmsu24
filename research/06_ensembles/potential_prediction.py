import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, X, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, X, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(X)

    def transform(self, X):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        X = X.copy()
        types = []

        for i in range(len(X)):
            min_val, max_val = np.min(X[i]), np.max(X[i])
            sizex, sizey = X[i].shape[1], X[i].shape[0]

            idx = np.where(X[i] < max_val)
            top, bottom = idx[0].min(), idx[0].max()
            left, right = idx[1].min(), idx[1].max()

            # Center potential
            offx = sizex // 2 - (left + right) // 2
            offy = sizey // 2 - (top + bottom) // 2

            if offx > 0:
                X[i] = np.hstack([max_val * np.ones(shape=(sizey, offx)), X[i][:, :sizex - offx]])
            elif offx < 0:
                X[i] = np.hstack([X[i][:, -offx:], max_val * np.ones(shape=(sizey, -offx))])

            if offy > 0:
                X[i] = np.vstack([max_val * np.ones(shape=(offy, sizex)), X[i][:sizey - offy, :]])
            elif offy < 0:
                X[i] = np.vstack([X[i][-offy:, :], max_val * np.ones(shape=(-offy, sizex))])

            assert X[i].shape == (sizey, sizex)

            # Detect potential type
            if np.where((X[i] < (min_val + max_val) * 2 / 3) & (X[i] > (min_val + max_val) / 3))[0].size > 20:
                types.append('harmonic')
            elif np.where(idx[0] == top)[0].size > 20 and np.where(idx[1] == left)[0].size > 20:
                types.append('rectangle')
            else:
                types.append('elliptic')

        onehot = OneHotEncoder().fit_transform(np.array(types)[:, None], np.ones(shape=(len(X),)))

        return np.hstack([X.reshape((X.shape[0], -1)), onehot.toarray()])
        # return X, types


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline([
        ('vectorizer', PotentialTransformer()),
        ('regressor', BaggingRegressor(RidgeCV(np.logspace(-5, 6, 11)), n_jobs=-1, n_estimators=100))
    ])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
