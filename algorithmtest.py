import os
import struct
import numpy as np
import pdb


import readdata as rd
import LinearRegression.linearreg as linear
import LogisticRegression.logisticreg as logistic
import tools

MNIST = 1

def readfile(fileindex):
    if fileindex == MNIST:
        with open('/home/lee/Documents/dataset/MNIST_data/t10k-labels.idx1-ubyte', 'rb') as f1:
            magic, num = struct.unpack(">II", f1.read(8))
            label_test = np.fromfile(f1, dtype=np.uint8)
        with open('/home/lee/Documents/dataset/MNIST_data/t10k-images.idx3-ubyte', 'rb') as f2:
            magic, num, rows, cols = struct.unpack(">IIII", f2.read(16))
            img_test = np.fromfile(f2, dtype=np.uint8).reshape(len(label_test), rows, cols)

        with open('/home/lee/Documents/dataset/MNIST_data/train-labels.idx1-ubyte', 'rb') as f3:
            magic, num = struct.unpack(">II", f3.read(8))
            label_train = np.fromfile(f3, dtype=np.uint8)
        with open('/home/lee/Documents/dataset/MNIST_data/train-images.idx3-ubyte', 'rb') as f4:
            magic, num, rows, cols = struct.unpack(">IIII", f4.read(16))
            img_train = np.fromfile(f4, dtype=np.uint8).reshape(len(label_train), rows, cols)
        return img_train, label_train, img_test, label_test

CLS = 1
REG = 0
if __name__ == "__main__":
    lr = 10
    iternum = 100
    method = 1
    threshold = 0.5
    if REG:
        data, label = rd.readcsv("./data/reg_train.csv")
        y = np.array([data[:, 0]])

        y = y.T
        top = np.max(y)
        bot = np.min(y)
        me = (top + bot) / 2
        y = (y - me) / (top - bot) * 2

        X = data[:, 3:5]
        top = np.max(X, axis=0)
        bot = np.min(X, axis=0)
        me = (top + bot) / 2
        X = (X - me) / (top - bot) * 2

        weight = linear.train(X, y, lr, method, iternum)
    if CLS:
        tag, data, label = rd.readcls("./data/cls_train.csv")
        label = np.array([label])
        label = label.astype(np.float)
        label = label.T
        X = np.array(data)
        X = X.astype(np.float)
        X = X[:, 0:48]

        weight = logistic.train(X, label, lr, method, iternum)

        test_tag, test_data, test_label = rd.readcls("./data/cls_test.csv")
        test_label = np.array([test_label])
        test_label = test_label.astype(np.float)
        test_label = test_label.T

        X_test = np.array(test_data)
        X_test = X_test.astype(np.float)
        X_test = X_test[:, 0:48]

        result = logistic.test(X_test, weight, threshold)
        acc = np.sum(result == test_label)/len(test_label)
        print(acc)

#    NUMBER = list(range(0, 10))
#    TRAIN = []
#    TEST = []
#    trainsize = 100
#    img_train, label_train, img_test, label_test = readfile(MNIST)
#    lt, mt, nt = img_train.shape
#    matrix_train = img_train.reshape(lt, mt * nt).transpose()
#    matrix_train = matrix_train.astype(np.float64)/255.0
#    label_train = np.array([label_train]).transpose()
#
#    lt, mt, nt = img_test.shape
#    matrix_test = img_test.reshape(lt, mt * nt).transpose()
#    matrix_test = matrix_test.astype(np.float64)/255.0
#    label_test = np.array([label_test]).transpose()
#
#    for i in NUMBER:
#        index = np.where(label_train == i)
#        tmp = matrix_train[:, index[0]]
#        TRAIN.append(tmp)
#    
#    for i in NUMBER:
#        index = np.where(label_test == i)
#        tmp = matrix_test[:, index[0]]
#        TEST.append(tmp)
#    
#    X = []
#    y = []
#    testset = [0, 1]
#    for i in testset:
#        tmp = TRAIN[i]
#
#        X += tmp[:, 0:trainsize]
#        y += trainsize * [i]
#    X = np.array(X)
#    y = np.array(y)
#
#
#    lr = 0.1
#    w = logis.train(X, y, lr, 1, 100)
#
#    testX = []
#    testy = []
#    testsize = 100
#    for i in testset:
#        tmp = TEST[i]
#        testX += tmp[:, 0:testsize]
#        testy += testsize * [i]
#    testX = np.array(testX)
#    testy = np.array(testy)
#    acc = logis.test(testX, testy, w)
#
#    print(acc)
#
