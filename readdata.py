import numpy as np
import pdb
import csv

def readcsv(filename):
    tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
    data = tmp[1:, 2:].astype(np.float)
    label = tmp[0, 2:].astype(np.str)

    return data, label

def readcls(filename):
    tag = []
    data = []
    label = []
    fp = open(filename, 'r')
    reader = csv.reader(fp)
    rows = [row for row in reader]
    fp.close()
    tag.append(rows[0])
    for i in range(1, len(rows)):
        label.append(rows[i][len(rows[i]) - 1])
        data.append(rows[i])
    return tag, data, label

