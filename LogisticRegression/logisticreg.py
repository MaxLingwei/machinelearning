import numpy as np
import pdb
import tools

def train(X, label, lr, method, itern):
    N, M = X.shape
    bias = np.ones(N)
    X = np.c_[bias, X]
    M = M + 1
    weight = np.random.rand(1, M) - np.random.rand(1, M)
    X = tools.normmatrix(X)
    weight = tools.normmatrix(weight)
    weight = weight.T

    if method == 1:
        for i in range(1, itern):
            result = p_1(X, weight)
            w_grad = np.dot(X.T, label - result)/N
            weight -= lr * w_grad

            pre = result > 0.5
            acc = np.sum(label == pre) / len(label)
            print("iter" + str(i) + "acc:" + str(acc))
        return weight
    else:
        a = 0

def test(X, weight, threshold):
    N, M = X.shape
    bias = np.ones(N)
    X = np.c_[bias, X]
    X = tools.normmatrix(X)
    result = p_1(X, weight)
    return (result > threshold)

def p_1(X, weight):
    tmp = np.exp(np.dot(X, weight))
    return tmp / (1 + tmp)

