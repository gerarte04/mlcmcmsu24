import numpy as np
import typing


class MinMaxScaler:
    _min_x, _max_x = np.array([]), np.array([])

    def fit(self, data: np.ndarray) -> None:
        self._min_x = np.min(data, axis=0)
        self._max_x = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._min_x) / (self._max_x - self._min_x)


class StandardScaler:
    _mean, _std = np.array([]), np.array([])

    def fit(self, data: np.ndarray) -> None:
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std
