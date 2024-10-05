import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    return X[::4, 120:500:5]

def sum_non_neg_diag(X: np.ndarray) -> int:
    diag = X.diagonal()
    nonneg_diag = diag[np.where(diag >= 0)]
    return nonneg_diag.sum() if nonneg_diag.size > 0 else -1

def replace_values(X: np.ndarray) -> np.ndarray:
    X1 = X.copy()
    return np.select([np.logical_or(X1 > 1.5 * X1.mean(axis=0), X1 < 0.25 * X1.mean(axis=0))], [-1], X1)
