import numpy as np

def normmatrix(A):
    veclen = np.linalg.norm(A, axis=1)
    vecmat = np.array([veclen])
    vecmat = vecmat.T
    return A / vecmat
