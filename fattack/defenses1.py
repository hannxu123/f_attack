
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from utils import *
from fair_methods.fair_funcs_const import fair_train_const


def log_logistic_torch(X):
    out = torch.empty_like(X)  # same dimensions and data types
    idx = X > 0
    out[idx] = -torch.log(1.0 + torch.exp(-X[idx]))
    out[~idx] = X[~idx] - torch.log(1.0 + torch.exp(X[~idx]))
    return out

def logistic_loss_torch(w, X, y):
    yz = y * torch.matmul(w, X)
    out = - torch.mean(log_logistic_torch(yz))
    return out


def no_defense(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, fair_constraint):
    ## Slab Defense
    print('.........No Defense.....................')
    ## fair classification
    tol_list = np.linspace(1.0, 0.00, 30)
    for j in range(30):
        w1 = fair_train_const(X_train1, y_train1, a_train1, tol_list[j])
        y_pred = np.sign(np.matmul(X_test, w1.T).flatten())   ## test performance
        result3 = test_fairness(y_pred, y_test, a_test)
        y_pred = np.sign(np.matmul(X_valid, w1.T).flatten())     ## valid performance
        result4 = test_fairness(y_pred, y_valid, a_valid)
        #print(result3, result4)
        if (result4[1] < fair_constraint):
            print('***', result3, flush= True)
            break
    print(result3, result4)


def kNN(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, fair_constraint):
    ## Slab Defense
    print('.........kNN Defense.....................')
    x_t = np.copy(X_train1)
    y_t = np.copy(y_train1)
    a_t = np.copy(a_train1)

    x_t1 = torch.tensor(x_t).cuda()
    y_t1 = torch.tensor(y_t).cuda()
    a_t1 = torch.tensor(a_t).cuda()

    ## kNN
    p_score = []
    for i in range(x_t.shape[0]):
        x_target = (x_t1[i])
        y_target = y_t[i]
        dist_vec = torch.diag((x_t1[y_t1 == y_target]- x_target) @ (x_t1[y_t1 == y_target] - x_target).T)
        p_score.append(dist_vec[6].item())

    p_score=  np.array(p_score)
    q = np.percentile(p_score, 80)
    benign_idx = (p_score < q)
    x_t = x_t[benign_idx]
    y_t = y_t[benign_idx]
    a_t = a_t[benign_idx]
    tol_list = np.linspace(1.0, 0.00, 30)
    for i in range(30):
        w = fair_train_const(x_t, y_t, a_t, tol_list[i])
        y_pred = np.sign(np.matmul(X_test, w.T).flatten())  ## test performance
        result3 = test_fairness(y_pred, y_test, a_test)
        y_pred = np.sign(np.matmul(X_valid, w.T).flatten())  ## valid performance
        result4 = test_fairness(y_pred, y_valid, a_valid)
        if (result4[1] < fair_constraint):
            print('***', result3, result4, flush=True)
            break
    print(result3, result4, flush=True)


def sever_train(w, x, y, bar):
    ## get the loss value
    w = torch.tensor(w).float()
    y = torch.tensor(y).float()
    w.requires_grad = True

    grad_sub = []
    for j in range(x.shape[0]):
        x_j = torch.tensor(x[j:j+1]).float()
        out_loss = - y[j:j+1] * w @ x_j.T
        w_grad = torch.autograd.grad(out_loss, w)[0]
        grad_sub.append(w_grad.cpu().numpy().flatten())

    grad_sub = np.array(grad_sub)
    normalzed_grad_sub = grad_sub - np.mean(grad_sub, axis=0)

    ## do the SVD decomposition
    pca = TruncatedSVD(n_components=1)
    pca.fit(normalzed_grad_sub)

    poison_score = np.square(normalzed_grad_sub.dot(pca.components_[0]))
    benign_idx = np.where(poison_score < np.percentile(poison_score, bar))[0]
    flag_idx = np.where(poison_score >= np.percentile(poison_score, bar))[0]
    return benign_idx, flag_idx


def sever(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, fair_constraint):
    print('.........SEVER Defense.....................')
    x_t = np.copy(X_train1)
    y_t = np.copy(y_train1)
    a_t = np.copy(a_train1)
    for j in range(15):
        print('** SEVER round ' +str(j))
        w = fair_train_const(x_t, y_t, a_t, 0.1)
        bengin_idx, _ = sever_train(w, x_t, y_t, 98)
        x_t = x_t[bengin_idx]
        y_t = y_t[bengin_idx]
        a_t = a_t[bengin_idx]
        tol_list = np.linspace(1.0, 0.00, 30)
        for i in range(30):
            w = fair_train_const(x_t, y_t, a_t, tol_list[i])
            y_pred = np.sign(np.matmul(X_test, w.T).flatten()) ## test performance
            result3 = test_fairness(y_pred, y_test, a_test)
            y_pred = np.sign(np.matmul(X_valid, w.T).flatten())  ## valid performance
            result4 = test_fairness(y_pred, y_valid, a_valid)
            #print('round ', result3, result4, flush= True)
            if (result4[1] < fair_constraint):
                print('*** ', result3, flush= True)
                break
        print('SEVER result ', result3, result4, flush=True)


