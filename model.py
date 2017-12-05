#!/usr/bin/env python


"""
"""


import numpy as np
from datahelper import load_quora, load_quora_train, load_quora_test
from sklearn import linear_model
from eval import log_csv
import csv


# load train
# x_train, y_train = load_quora('Quora-QP_train_new.txt')
x_train, y_train = load_quora_train('../com/train.csv')

# define model
lrg = linear_model.LogisticRegression()

# train a model
lrg.fit(x_train, y_train)

# load text data
# x_test, y_test = load_quora('Quora-QP_test_new.txt')
x_test = load_quora_test('../com/test.csv')

# predict the prob of x_text
y_predict = lrg.predict_proba(x_test)
y_predict_self = lrg.predict_proba(x_train)

assert y_predict.shape[1] == 2, 'y_predict is not right'

# write csv
with open('result1.csv', 'w') as  csvfile:
    fieldnames = ['label', 'predict']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    size = y_train.shape[0]
    for i in range(size):
        writer.writerow({'label':y_train[i], 'predict':y_predict_self[i][1]})

# log loss
print(log_csv('result1.csv'))

# write kaggle text
with open('sub1.csv', 'w') as subfile:
    fieldnames = ['test_id', 'is_duplicate']
    writer = csv.DictWriter(subfile, fieldnames=fieldnames)
    size = y_predict.shape[0]
    writer.writeheader()
    for i in range(size):
        writer.writerow({'test_id':i, 'is_duplicate':y_predict[i][1]})
