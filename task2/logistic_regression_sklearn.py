#   coding: utf-8
#   This file is part of WatermelonBookFromDeepshare.

#   WatermelonBookFromDeepshare is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License.

__author__ = 'amk'
__version__ = 1.0
__maintainer__ = 'amk'
__email__ = "alienmarkwong@163.com"
__date__ = "2020/06/25"

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

print(__doc__)

t0 = time.time()
train_sample = 5000
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(x.shape[0])

x = x[permutation]
y = y[permutation]
x = x.reshape((x.shape[0], -1))

x_train ,x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, test_size=10000)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = LogisticRegression(C=50.0/train_sample,
                         multi_class='multinomial',
                         penalty='l1',
                         solver='saga',
                         tol=0.1)
clf.fit(x_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(x_test, y_test)

print('Sparsity with L1: %.2f%%' % sparsity)
print('Test score with L1: %.4f' % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()

for i in range(10):
    l1_plot = plt.subplot(2, 5, i+1)
    l1_plot.imshow(coef[i].reshape(28, 28),
                   interpolation='nearest',
                   cmap=plt.cm.RdBu,
                   vmin=-scale,
                   vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i'%i)

plt.suptitle("classification vector for ...")

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()


