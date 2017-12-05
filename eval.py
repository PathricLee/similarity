#!/usr/bin/env python
#_*_ coding:utf-8_*_


"""
eval a result. p, r, f
"""


import math
import csv


def run(txt):
    """ same / all """
    same = 0
    all = 0
    for line in open(txt):
        line = line.strip()
        segs = line.split('\t')
        label = segs[0]
        predict = segs[-1]
        all += 1
        if label == predict:
            same += 1
    recall = float(same) / all
    print(recall)

def log(txt):
    """ -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp)) """
    result = 0
    all = 0
    epsilon = 1e-15
    for line in open(txt):
        line = line.strip()
        segs = line.split('\t')
        label = float(segs[0])
        predict = max(epsilon, float(segs[-2]))
        predict = min(predict, 1 - epsilon)
        result -= label * math.log(predict) + (1 - label) * math.log(1 - predict)
        all += 1
    print(result, all, result / all)


def log_csv(txt):
    """ -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp)) """
    result = 0
    all = 0
    epsilon = 1e-15
    _csv = open(txt, 'r')
    reader = csv.reader(_csv)
    for line in reader:
        label, predict = line 
        label = float(label)
        predict = max(epsilon, float(predict))
        predict = min(predict, 1 - epsilon)
        result -= label * math.log(predict) + (1 - label) * math.log(1 - predict)
        all += 1
    print(result, all, result / all)

if __name__ == '__main__':
    # run('tmp')
    log('tmp')
