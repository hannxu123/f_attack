
import numpy as np
from random import seed, shuffle
import loss_funcs as lf # our implementation of loss funcs
from scipy.optimize import minimize # for loss func minimization
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt # for plotting stuff
import sys
import copy
import time
import warnings
import numpy as np
from scipy.optimize import minimize, rosen

def log_logistic(X):
    out = np.empty_like(X)  # same dimensions and data types
    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

def logistic_loss(w, X, y):
    yz = y * X.dot(w)
    out = - np.sum(log_logistic(yz))
    return out


def fair_train_cov(x_train, y_train, x_control_train, thresh):
    max_iter = 20  # maximum number of iterations for the minimization algorithm
    constraint0 = ({'type': 'ineq', 'fun': cov_tp_constraint,
              'args': (x_train, y_train, x_control_train, thresh)})


    f_args = (x_train, y_train)
    w = minimize(fun=logistic_loss,
                 x0=np.random.rand(x_train.shape[1], ),
                 args=f_args,
                 method='SLSQP',
                 options={"maxiter": max_iter},
                 constraints=(constraint0),
                 )
    weight = w['x']
    return weight

def cov_tp_constraint(model, x_arr, y_arr, x_control, thresh):

    if model is None:
        arr = y_arr  # simply the output labels
    else:
        #arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label
        arr = x_arr.dot(model)

    pos_class = (y_arr == 1)
    arr1 = np.array(arr, dtype=np.float)[pos_class]
    cov1 = np.dot(x_control[pos_class] - np.mean(x_control[pos_class]), (arr1)) / float(len(x_control[pos_class]))
    ans = thresh - abs(cov1)
    return ans