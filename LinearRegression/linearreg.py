import numpy as np
import pdb

def train(X, y, lr, method, itern):
    N, M = X.shape
    weight = np.random.rand(M, 1)
    bias = np.ones([N, 1])

    if method == 1:
        for i in range(1, itern):
            pre = np.dot(X, weight) + bias
            loss = mse(pre, y)
            w_grad = 2 / N * np.dot(X.T, pre- y)
            b_grad = 2 / N * np.sum(pre - y)
            weight -= lr * w_grad
            bias -= lr * b_grad * np.ones([N, 1])

            print("iter" + str(i) + "loss:" + str(loss))
        return weight

    else:
        a = 0
def mse(pre, y):
    return ((pre - y) ** 2).mean(axis=0)


def printerr(err):
    print(np.sum(err))

def test(X, y, weight):
    threshold = 0.5
    err = np.exp(np.dot(X.transpose(), weight))
    err = np.true_divide(err, 1 + err)
    label_eval = (err > threshold) + 0
    return np.sum(label_eval == y) / len(y)
