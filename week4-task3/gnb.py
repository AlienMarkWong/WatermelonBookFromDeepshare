#   coding: utf-8
#   This file is part of WatermelonBookFromDeepshare.

#   WatermelonBookFromDeepshare is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License.

__author__ = 'amk'
__version__ = 1.0
__maintainer__ = 'amk'
__email__ = "alienmarkwong@163.com"
__date__ = "2020/07/18"

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
# Number of mislabeled points out of a total 75 points : 4