
import numpy as np
from scipy.optimize import minimize # for loss func minimization

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


def fair_train_const(x_train, y_train, x_control_train, thresh):
    max_iter = 1000  # maximum number of iterations for the minimization algorithm
    constraint0 = ({'type': 'ineq', 'fun': const_tp_constraint,
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


def const_tp_constraint(model, x_arr, y_arr, x_control, thresh):

    if model is None:
        arr = y_arr  # simply the output labels
    else:
        arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label

    ## use subsitute loss functions
    logit = - log_logistic(arr) * y_arr
    #logit = - arr

    loss_prot = logit[x_control == 1]
    y_prot = y_arr[x_control == 1]
    tpr_prot = np.sum(loss_prot * (y_prot == 1)) / np.sum(y_prot == 1)

    loss_unprot = logit[x_control == 0]
    y_unprot = y_arr[x_control == 0]
    tpr_unprot = np.sum(loss_unprot * (y_unprot == 1)) / np.sum(y_unprot == 1)

    ans = thresh - abs(tpr_unprot - tpr_prot)

    return ans

