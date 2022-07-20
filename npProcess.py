# -*- coding: utf-8 -*-
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
import pydot
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle
import tkinter as tk

def Z_ScoreNormalization(x, mu, sigma):
    xx = (x - mu) / sigma
    return xx


from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import functools
import itertools
import numpy
import operator
# import perfplot
from collections import Iterable  # or from collections.abc import Iterable
from sklearn.cluster import KMeans


# from iteration_utilities import deepflatten

# 使用两次for循环
def forfor(a):
    return [item for sublist in a for item in sublist]


# 通过sum
def sum_brackets(a):
    return sum(a, [])


# 使用functools內建模块
def functools_reduce(a):
    return functools.reduce(operator.concat, a)


# 使用itertools內建模块
def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))


# 使用numpy
def numpy_flat(a):
    return list(numpy.array(a).flat)


# 使用numpy
def numpy_concatenate(a):
    return list(numpy.concatenate(a))


# 自定义函数
def flatten(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def pylangs_flatten(a):
    return list(flatten(a))


with open('UWB/UWB.data', 'rb') as f:
    dataU = pickle.load(f)


yU1 = loadmat('UWB/lableY.mat')['data']
yyU1 = loadmat('UWB/lableYY.mat')['data']

bb = np.zeros(([len(dataU), 8])) #
for i in range(len(dataU)):
    bb[i, :] = (np.array(dataU[i]))
xArUWB = bb
print((xArUWB.shape))





xtrain, xtest, ytrain, ytest = \
    model_selection.train_test_split(xArUWB, yU1.T, test_size=0.2)

print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

print(len(xtrain))
print('测试集：%s' % len(xtest))
clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_leaf_nodes=500)


clf.fit(xtrain, ytrain.ravel())


def calculate_result(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='weighted')
    m_recall = metrics.recall_score(actual, pred, average='weighted')
    print('predict info:')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred, average='weighted')))



ytrain_pred = clf.predict(xtrain)

calculate_result(ytrain.ravel(), ytrain_pred)
print('训练集的准确率：%s' % accuracy_score(y_true=ytrain, y_pred=ytrain_pred))
ytest_pred = clf.predict(xtest)
#ytest_pred = clf.predict(xtest[:, :-1])
print('测试集的准确率：%s' % accuracy_score(y_true=ytest, y_pred=ytest_pred))

labels = ['1','2','3','4','5','6','7','8']
tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(ytest, ytest_pred)

np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 8), dpi=120)


ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):

    c = cm_normalized[y_val][x_val]

    if (c > 0.01):
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()

