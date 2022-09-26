
import torch
from argparse import Namespace
from roh.fr_roh import train_model

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wang.dro_training as dro_training
import pandas as pd

def DRO(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, fair_constraint):
    def make_df(X, y, a):
        df = pd.DataFrame([])
        names = []
        for i in range(X.shape[1]):
            x_vec = X[:, i]
            name = str(i)
            df[name] = x_vec
            names.append(name)
        df['y'] = y
        df['a0'] = a
        return df, names

    ############## wang
    print('...................do fairness training using DRO......................', flush=True)
    PROTECTED_COLUMNS = ['a0']
    LABEL_COLUMN = 'y'
    PROXY_COLUMNS = PROTECTED_COLUMNS

    df, FEATURE_NAMES  = make_df(X_train1, y_train1, a_train1)
    df_val, _ = make_df(X_valid, y_valid, a_valid)
    df_test, _ = make_df(X_test, y_test, a_test)

    dro_training.get_results_for_learning_rates(df, df_val, df_test, FEATURE_NAMES,
                PROTECTED_COLUMNS, PROXY_COLUMNS, LABEL_COLUMN, constraint='tpr', constraints_slack = fair_constraint)


def roh(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, fair_constraint):
    ## dataset

    x_train = torch.FloatTensor(X_train1).cuda()
    y_train = torch.FloatTensor(y_train1).cuda()
    a_train = torch.FloatTensor(a_train1).cuda()

    x_test = torch.FloatTensor(X_test).cuda()
    y_test = torch.FloatTensor(y_test).cuda()
    a_test = torch.FloatTensor(a_test).cuda()

    x_val = torch.FloatTensor(X_valid).cuda()
    y_val = torch.FloatTensor(y_valid).cuda()
    a_val = torch.FloatTensor(a_valid).cuda()

    train_result = []
    train_tensors = Namespace(XS_train=x_train, y_train=y_train, s1_train=a_train)
    val_tensors = Namespace(XS_val=x_val, y_val=y_val, s1_val=a_val)
    test_tensors = Namespace(XS_test=x_test, y_test=y_test, s1_test=a_test)

    train_opt = Namespace(val=len(y_val), n_epochs=10000, k=5, lr_g=0.001, lr_f=0.001, lr_r=0.001)
    seed = 1

    lambda_f_set = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.52]
    lambda_r = 0.4
    for lambda_f in lambda_f_set:
        train_result.append(train_model(train_tensors, val_tensors, test_tensors, train_opt,
                                         lambda_f=lambda_f, lambda_r=lambda_r, seed=seed))

