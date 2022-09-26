import torch
from sklearn.decomposition import TruncatedSVD
from utils import *
from fair_methods.fair_funcs_const import fair_train_const
from sklearn.linear_model import LogisticRegression
import torch.nn as nn


def rfc_train(x, y, rate):
    ## get the loss value
    all_benign_idx = []

    for i in range(4):
        x_subgroup = np.where(y == i)[0]
        xx = x[x_subgroup]
        center_xx = xx - np.mean(xx, axis=0)

        pca = TruncatedSVD(n_components=1)
        pca.fit(center_xx)
        d0 = pca.components_[0].reshape(x.shape[1], 1)  ## top right singular vector
        ip = (center_xx @ d0).flatten()

        benign_idx1 = np.ones(x.shape[0])
        idx1 = np.where(y == i)[0]
        bad_set1 = idx1[ip < np.percentile(ip, rate)]
        benign_idx1[bad_set1] = 0
        all_benign_idx.append(benign_idx1)

        benign_idx2 = np.ones(x.shape[0])
        idx2 = np.where(y == i)[0]
        bad_set2 = idx2[ip > np.percentile(ip, rate)]
        benign_idx2[bad_set2] = 0
        all_benign_idx.append(benign_idx2)

    all_benign_idx = np.array(all_benign_idx)
    return all_benign_idx


def rfc(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, fair_constraint, rfc_rate):
    print('.........RFC Defense.....................')
    x_t = np.copy(X_train1)
    y_t = np.copy(y_train1)
    a_t = np.copy(a_train1)
    y_a = (y_t + 1) + a_t

    for j in range(20):
        print('--- RFC round ' +str(j))
        bengin_idx = rfc_train(x_t, y_a, rfc_rate)

        result = []
        for i in range(bengin_idx.shape[0]):
            x_t1 = x_t[bengin_idx[i] == 1]
            y_t1 = y_t[bengin_idx[i] == 1]
            a_t1 = a_t[bengin_idx[i] == 1]

            tol_list = np.linspace(1.5, 0.00, 30)
            val_perform = []
            for k in range(30):
                w = fair_train_const(x_t1, y_t1, a_t1, tol_list[k])
                y_pred = np.sign(np.matmul(X_valid, w.T).flatten())  ## valid performance
                result4 = test_fairness(y_pred, y_valid, a_valid)
                acc = result4[0]
                unfair = result4[1]
                valid_performance = acc - 3 * ((unfair - fair_constraint) > 0) * (unfair - fair_constraint)
                val_perform.append(valid_performance)
            best_val_perform = np.max(np.array(val_perform))
            print('.......Group ', i, best_val_perform)
            result.append(best_val_perform)
        ws = np.argmax(np.array(result))    ## worst set

        print('choose the group ' +str(ws), flush = True)
        x_t = x_t[bengin_idx[ws] == 1]
        y_t = y_t[bengin_idx[ws] == 1]
        a_t = a_t[bengin_idx[ws] == 1]
        y_a = y_a[bengin_idx[ws] == 1]
        tol_list = np.linspace(1.5, 0.00, 40)
        for i in range(40):
            w = fair_train_const(x_t, y_t, a_t, tol_list[i])
            y_pred = np.sign(np.matmul(X_test, w.T).flatten()) ## test performance
            result3 = test_fairness(y_pred, y_test, a_test)
            y_pred = np.sign(np.matmul(X_valid, w.T).flatten())  ## valid performance
            result4 = test_fairness(y_pred, y_valid, a_valid)
            if i % 5 == 0:
                print(result3, result4, flush= True)
            if (result4[1] < fair_constraint):
                print('*************************** RFC Result ', result3, flush= True)
                break
        print('*************************** RFC Result ', result3, flush=True)


