#!/usr/bin/env python


"""
"""


import numpy as np
from sklearn import linear_model, datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x, y)

y_predict = logreg.predict(x)
print(y_predict[0:100])
print(y[0:100])
