import numpy as np
import pandas as pd
import os
import argparse
from dataset_adult import *
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
from utils import *
from defenses1 import sever, no_defense, kNN

from fair_methods.fair_funcs_const import fair_train_const

def main(args):

    ## get train/test data
    X_train, X_test, X_valid, y_train, y_test, y_valid, a_train, a_test, a_valid = preprocess_adult_data(seed = args.seed)
    #clf = LogisticRegression(fit_intercept=False).fit(X_train, y_train)    ## logistic regression
    #pred = clf.predict(X_test)
    #result4 = test_fairness(pred, y_test, a_test)
    print('1, 1 ',np.sum((y_train == 1) & (a_train == 1)) / X_train.shape[0])
    print('1, 0',np.sum((y_train == 1) & (a_train == 0)) / X_train.shape[0])
    print('-1, 1',np.sum((y_train == -1) & (a_train == 1)) / X_train.shape[0])
    print('-1, 0',np.sum((y_train == -1) & (a_train == 0)) / X_train.shape[0])

    ## attack methods
    number = int(X_train.shape[0] * args.rate)
    print('************Generate Poisoning Samples*****************')
    if args.attack_method == 'label_flip':
        print('Doing Label Flipping Attack')
        attack_set = np.random.choice(np.where(y_train == - 1)[0], number, replace=False)
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        y_train1[attack_set] = -y_train1[attack_set]
    elif args.attack_method == 'attr_flip':
        print('Doing Attr. Flipping Attack')
        attack_set = np.random.choice(X_train.shape[0], number, replace=False)
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        a_train1[attack_set] = 1-a_train1[attack_set]
    elif args.attack_method == 'mm1':
        print('Doing Min-Max Attack with Z = ' + str(1))
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        from mm_attack import cert_attack
        clf = LogisticRegression(fit_intercept=False).fit(X_train, y_train)
        weight = clf.coef_
        X_new, y_new = cert_attack(X_train1, y_train1, weight, int(X_train1.shape[0] * args.rate), args.bar)
        a_new = 1 * np.ones(y_new.shape[0])
        X_train1 = np.concatenate([X_new, X_train1], axis=0)
        y_train1 = np.concatenate([y_new, y_train1], axis=0)
        a_train1 = np.concatenate([a_new, a_train1], axis=0)
    elif args.attack_method == 'mm0':
        print('Doing Min-Max Attack with Z = ' + str(0))
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        from mm_attack import cert_attack
        clf = LogisticRegression(fit_intercept=False).fit(X_train, y_train)
        weight = clf.coef_
        X_new, y_new = cert_attack(X_train1, y_train1, weight, int(X_train1.shape[0] * args.rate), args.bar)
        a_new = 0 * np.ones(y_new.shape[0])
        X_train1 = np.concatenate([X_new, X_train1], axis=0)
        y_train1 = np.concatenate([y_new, y_train1], axis=0)
        a_train1 = np.concatenate([a_new, a_train1], axis=0)
    elif args.attack_method == 'f_attack1':
        print('Doing F-Attack with Z = ' + str(1))
        from f_attack1 import cert_fair_attack
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        w = fair_train_const(X_train, y_train, a_train, args.fair_constraint)
        X_new, y_new, a_new = cert_fair_attack(X_train1, y_train1, a_train1, w,
                                            int(X_train1.shape[0] * args.rate), 1, args.bar)
        X_train1 = np.concatenate([X_new, X_train1], axis=0)
        y_train1 = np.concatenate([y_new, y_train1], axis=0)
        a_train1 = np.concatenate([a_new, a_train1], axis=0)
    elif args.attack_method == 'f_attack0':
        print('Doing F-Attack with Z = ' + str(0))
        from f_attack1 import cert_fair_attack
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        w = fair_train_const(X_train, y_train, a_train, args.fair_constraint)
        X_new, y_new, a_new = cert_fair_attack(X_train1, y_train1, a_train1, w,
                                            int(X_train1.shape[0] * args.rate), 0, args.bar)
        X_train1 = np.concatenate([X_new, X_train1], axis=0)
        y_train1 = np.concatenate([y_new, y_train1], axis=0)
        a_train1 = np.concatenate([a_new, a_train1], axis=0)
    elif args.attack_method == 'f_attacks':
        print('Doing F*-Attack')
        from f_attack2 import cert_fair_attack
        X_train1 = np.copy(X_train)
        y_train1 = np.copy(y_train)
        a_train1 = np.copy(a_train)
        w = fair_train_const(X_train, y_train, a_train, args.fair_constraint)
        X_new, y_new, a_new = cert_fair_attack(X_train1, y_train1, a_train1, w, int(X_train1.shape[0] * args.rate), args.fair_constraint, args.bar)
        X_train1 = np.concatenate([X_new, X_train1], axis=0)
        y_train1 = np.concatenate([y_new, y_train1], axis=0)
        a_train1 = np.concatenate([a_new, a_train1], axis=0)
    else:
        raise ValueError

    if args.defenses == 1:
        ## Sphere
        no_defense(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, args.fair_constraint)

        ## KNN Defense
        kNN(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, args.fair_constraint)

        ## SEVER DEFENSE
        sever(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, args.fair_constraint)

    elif args.defenses == 2:
        ## RFC training
        from def_rfc3 import rfc
        if args.rate <= 0.1:
            rfc_rate = int(args.rate * 100)
        else:
            rfc_rate = int(args.rate * 200)
        rfc(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid,
            args.fair_constraint, rfc_rate)
    elif args.defenses == 3:
        from defenses2 import DRO
        DRO(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, args.fair_constraint)
    elif args.defenses == 4:
        from defenses2 import roh
        roh(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, args.fair_constraint)
    elif args.defenses == 5:
        ## RFC training
        from def_rfc4 import rfc
        a_num = X_new.shape[0]
        print(a_num)
        rfc(X_train1, y_train1, a_train1, X_test, y_test, a_test, X_valid, y_valid, a_valid, args.fair_constraint, a_num)
    else:
        raise ValueError


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--rate', type=float, default= 0.1)
    argparser.add_argument('--bar', type=float, default= 4.0)
    argparser.add_argument('--attack_method', type=str, default= 'f_attack')
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--fair_constraint', type=float, default= 0.05)
    argparser.add_argument('--defenses', type=int, default= 1)
    args = argparser.parse_args()
    print(args)
    main(args)