from typing import List
from copy import deepcopy

def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    X1 = deepcopy(X)
    X1 = [X1[i] for i in range(0, len(X1), 4)]

    for i in range(len(X1)):
        X1[i] = [X1[i][j] for j in range(120, 500, 5)]

    return X1

def sum_non_neg_diag(X: List[List[int]]) -> int:
    sum = 0
    have_nonneg = False

    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            have_nonneg = True
            sum += X[i][i]

    return sum if have_nonneg else -1

def replace_values(X: List[List[float]]) -> List[List[float]]:
    X1 = deepcopy(X)
    means = [0 for _ in range(len(X1[0]))]

    for j in range(len(X1[0])):
        for i in range(len(X1)):
            means[j] += X1[i][j]

    means = [means[i] / len(X1) for i in range(len(means))]

    for j in range(len(X1[0])):
        for i in range(len(X1)):
            if X1[i][j] < 0.25 * means[j] or X1[i][j] > 1.5 * means[j]:
                X1[i][j] = -1

    return X1
