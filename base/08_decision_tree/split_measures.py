import numpy as np


def evaluate_measures(sample):
    _, proba = np.unique(sample, return_counts=True)
    proba = proba.astype(np.float64) / len(sample)

    gini = np.sum(proba * (1 - proba))
    entropy = -np.sum(proba * np.log(proba))
    error = 1 - proba.max()

    measures = {'gini': float(gini), 'entropy': float(entropy), 'error': float(error)}
    return measures
