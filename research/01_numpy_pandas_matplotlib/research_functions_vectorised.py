import numpy as np
import pickle

def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.all(np.sort(x) == np.sort(y))

def max_prod_mod_3(x: np.ndarray) -> int:
    y, z = x[1:], x[:-1]
    mul = (y * z)[np.where(np.logical_or(y % 3 == 0, z % 3 == 0))]
    return mul.max() if mul.size > 0 else -1

def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (image * weights).sum(axis=2)

def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x1, y1 = x.transpose(), y.transpose()
    x1, y1 = np.repeat(x1[0], x1[1]), np.repeat(y1[0], y1[1])
    return np.dot(x1, y1) if x1.size == y1.size else -1

def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n1 = np.linalg.norm(X, axis=1)[:, None]
    n2 = np.linalg.norm(Y, axis=1).transpose()
    mtx = (X @ Y.transpose()).astype(np.float64)
    mtx /= n1
    mtx /= n2
    mtx = np.nan_to_num(mtx, nan=1)

    return mtx
