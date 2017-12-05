#!/usr/bin/env python


"""
support data api
"""


import csv
import similarity
import numpy as np


def load_quora(csv_file):
    _csv = open(csv_file, 'r')
    reader = csv.reader(_csv, delimiter='\t')
    y_labels = []
    x_features = []
    for line in reader:
        # no head-line
        feature = []
        label, ques1, ques2 = line
        ed, lcs, interrogative_lst = similarity.feature(ques1, ques2)
        feature.append(ed)
        feature.append(lcs)
        feature.extend(interrogative_lst)
        x_features.append(feature)
        y_labels.append(float(label))
    return np.array(x_features), np.array(y_labels)

    
def load_quora_train(csv_file):
    _csv = open(csv_file, 'r')
    reader = csv.reader(_csv)
    y_labels = []
    x_features = []
    for i, line in enumerate(reader):
        # head-line
        if i == 0:
            continue
        feature = []
        id_, q1_id, q2_id, ques1, ques2, label = line
        ed, lcs, interrogative_lst = similarity.feature(ques1, ques2)
        feature.append(ed)
        feature.append(lcs)
        feature.extend(interrogative_lst)
        x_features.append(feature)
        y_labels.append(float(label))
    return np.array(x_features), np.array(y_labels)


def load_quora_test(csv_file):
    _csv = open(csv_file, 'r')
    reader = csv.reader(_csv)
    x_features = []
    for i, line in enumerate(reader):
        # head-line
        if i == 0:
            continue
        feature = []
        id_, ques1, ques2 = line
        ed, lcs, interrogative_lst = similarity.feature(ques1, ques2)
        feature.append(ed)
        feature.append(lcs)
        feature.extend(interrogative_lst)
        x_features.append(feature)
    return np.array(x_features)

if __name__ == '__main__':
    x_dev, y_dev = load_quora('Quora-QP_dev_new.txt')
