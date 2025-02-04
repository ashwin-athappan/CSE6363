import numpy as np

def shuffle(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]