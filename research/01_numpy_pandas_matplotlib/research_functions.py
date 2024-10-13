from collections import Counter
from typing import List
import math

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    if len(x) != len(y):
        return False

    x1 = sorted(x)
    y1 = sorted(y)

    for i in range(len(x1)):
        if x1[i] != y1[i]:
            return False

    return True

def max_prod_mod_3(x: List[int]) -> int:
    if len(x) <= 1:
        return -1
    
    found = False
    max_mul = 0

    for i in range(1, len(x)):
        if (x[i] % 3 == 0 or x[i - 1] % 3 == 0) and (mul := x[i] * x[i - 1]) > max_mul:
            max_mul = mul
            found = True
    
    return max_mul if found else -1

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    mtx = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]

    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(weights)):
                mtx[i][j] += image[i][j][k] * weights[k]

    return mtx

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    x1, y1 = [], []

    for t in x:
        for _ in range(t[1]):
            x1.append(t[0])

    for t in y:
        for _ in range(t[1]):
            y1.append(t[0])

    if len(x1) != len(y1):
        return -1
    
    dot = 0

    for i in range(len(x1)):
        dot += x1[i] * y1[i]

    return dot

def vlen(x):
    sq_sum = 0

    for i in range(len(x)):
        sq_sum += x[i] * x[i]
    
    return math.sqrt(sq_sum)

def dot(x, y):
    dot = 0

    for i in range(len(x)):
        dot += x[i] * y[i]

    return dot

def iszero(x):
    for xi in x:
        if xi != 0:
            return False
    
    return True

def get_angle(x, y):
    if iszero(x) or iszero(y):
        return 1

    return dot(x, y) / (vlen(x) * vlen(y))

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    return [[get_angle(X[i], Y[j]) for j in range(len(Y))] for i in range(len(X))]
