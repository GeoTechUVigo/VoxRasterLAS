import numpy as np


# Variance faster than np.var()
def var(a: np.ndarray, axis: int = 0):
    return np.sum(abs(a - (a.sum(axis=axis) / len(a))) ** 2, axis=axis) / len(a)

