import numpy as np
import pandas as pd
import os
import argparse
from dataset_adult import *
from utils import *
import argparse
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class classifier1(torch.nn.Module):
    def __init__(self, input_shape):
        super(classifier1, self).__init__()
        self.linear1 = torch.nn.Linear(input_shape, 10)
        self.linear2 = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        out = self.linear2(x)
        return out, x

class classifier2(torch.nn.Module):
    def __init__(self):
        super(classifier2, self).__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        out = self.linear(x)
        return out


def evaluate(model1, X_test):
    device = 'cuda'
    N = X_test.shape[0]
    steps = np.floor(N / 100) + 1

    all_pred = []
    with torch.no_grad():
        for i in range(int(steps)):
            data = X_test[i * 100: (i+1) * 100]
            data = torch.tensor(data, dtype = torch.float).to(device)

            output, _ = model1(data)
            pred = torch.argmax(output, dim = 1)

            all_pred.append(pred.cpu().numpy())
    all_pred = np.concatenate(all_pred)
    return  all_pred

def clean_train(model1, X_train, y_train, optimizer1):

    device = 'cuda'
    N = X_train.shape[0]

    for i in range(200):

        idx = np.random.choice(N, 64, replace= False)
        data = X_train[idx]
        target = y_train[idx]

        data, target = torch.tensor(data, dtype = torch.float).to(device), \
                              torch.tensor(target, dtype = torch.long).to(device)
        #optimizer.zero_grad()
        logits, features = model1(data)

        ## classification loss
        optimizer1.zero_grad()
        loss1 = F.cross_entropy(logits, target)
        loss1.backward()
        optimizer1.step()



def adv_debias(model1, model2, X_train, y_train, a_train, optimizer1, optimizer2, alpha):

    device = 'cuda'
    N = X_train.shape[0]

    for i in range(200):

        idx = np.random.choice(N, 64, replace= False)
        data = X_train[idx]
        target = y_train[idx]
        attr = a_train[idx]

        data, target, attr = torch.tensor(data, dtype = torch.float).to(device), \
                              torch.tensor(target, dtype = torch.long).to(device), \
                             torch.tensor(attr, dtype = torch.long).to(device)

        #optimizer.zero_grad()
        logits, features = model1(data)

        ## sensitive loss
        optimizer2.zero_grad()
        loss2 = F.cross_entropy(model2(features), attr)
        loss2.backward(retain_graph = True)
        optimizer2.step()

        ## classification loss
        optimizer1.zero_grad()
        loss2 = F.cross_entropy(model2(features), attr)
        loss1 = F.cross_entropy(logits, target) - alpha * loss2
        loss1.backward()
        optimizer1.step()



def pgd_attack(model1, X1,
               X_train, y_train,
               attack_num,
               cls,
               epsilon = 9,
               num_steps = 30,
               step_size = 0.5):

    ## training from scratch
    optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    N = X_train.shape[0]
    X1 = torch.unsqueeze(torch.tensor(X1).to('cuda').float(), 0)

    all_adv_samples = []
    all_adv_labels = []

    for j in range(attack_num):

        idx = np.random.choice(N, 512, replace= False)
        data = X_train[idx]
        target = y_train[idx]
        data, target = torch.tensor(data, dtype = torch.float).to('cuda'), torch.tensor(target, dtype = torch.long).to('cuda')

        ## to maximize inner loss
        X_pgd = torch.clone(X1)
        X_pgd.requires_grad = True

        for i in range(num_steps):

            logits, features = model1(X_pgd)
            loss = F.cross_entropy(logits, torch.tensor([cls], dtype = torch.long).cuda())
            loss.backward()

            eta = step_size * X_pgd.grad.data
            X_pgd = X_pgd + eta

            if torch.norm(X_pgd.data - X1.data) > epsilon:
                X_pgd = X1 + (X_pgd.data - X1.data) / torch.norm(X_pgd.data - X1.data) * epsilon

            X_pgd = X_pgd.detach()
            X_pgd.requires_grad_()
            X_pgd.retain_grad()

        all_adv_samples.append(X_pgd.detach().cpu().numpy())
        all_adv_labels.append(cls)

        data = torch.cat([data, X_pgd.repeat(100, 1)], dim=0)
        target = torch.cat([target, cls * torch.ones(100, dtype=torch.long).cuda()])

        logits, features = model1(data)
        optimizer.zero_grad()
        loss1 = F.cross_entropy(logits, target)
        loss1.backward()
        optimizer.step()
        if j % 100 == 0:
            print(loss, loss1)

    all_adv_samples = np.concatenate(all_adv_samples, axis= 0)
    all_adv_labels = np.array(all_adv_labels)

    return all_adv_samples, all_adv_labels



def pgd_attack_fair(model1, model2, X1,
               X_train, y_train, a_train,
               attack_num,
               alpha2,
               cls, cla,
               epsilon = 9,
               num_steps = 30,
               step_size = 0.5):

    ## training from scratch
    optimizer1 = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    N = X_train.shape[0]
    X1 = torch.unsqueeze(torch.tensor(X1).to('cuda').float(), 0)
    device = 'cuda'

    all_adv_samples = []
    all_adv_labels = []
    all_adv_attr = []

    for j in range(attack_num):

        idx = np.random.choice(N, 512, replace= False)
        data = X_train[idx]
        target = y_train[idx]
        attr = a_train[idx]

        data, target, attr = torch.tensor(data, dtype = torch.float).to(device), \
                              torch.tensor(target, dtype = torch.long).to(device), \
                             torch.tensor(attr, dtype = torch.long).to(device)

        ## to maximize inner loss
        X_pgd = torch.clone(X1)
        X_pgd.requires_grad = True

        for i in range(num_steps):

            logits, features = model1(X_pgd)
            loss = F.cross_entropy(logits, torch.tensor([cls], dtype = torch.long).cuda())
            loss.backward()

            eta = step_size * X_pgd.grad.data
            X_pgd = X_pgd + eta

            if torch.norm(X_pgd.data - X1.data) > epsilon:
                X_pgd = X1 + (X_pgd.data - X1.data) / torch.norm(X_pgd.data - X1.data) * epsilon

            X_pgd = X_pgd.detach()
            X_pgd.requires_grad_()
            X_pgd.retain_grad()

        all_adv_samples.append(X_pgd.detach().cpu().numpy())
        all_adv_labels.append(cls)
        all_adv_attr.append(cla)

        data = torch.cat([data, X_pgd.repeat(100, 1)], dim=0)
        target = torch.cat([target, cls * torch.ones(100, dtype=torch.long).cuda()])
        attr = torch.cat([attr, cla * torch.ones(100, dtype=torch.long).cuda()])

        #optimizer.zero_grad()
        logits, features = model1(data)

        ## sensitive loss
        optimizer2.zero_grad()
        loss2 = F.cross_entropy(model2(features), attr)
        loss2.backward(retain_graph = True)
        optimizer2.step()

        ## classification loss
        optimizer1.zero_grad()
        loss2 = F.cross_entropy(model2(features), attr)
        loss1 = F.cross_entropy(logits, target) - alpha2 * loss2
        loss1.backward()
        optimizer1.step()

    all_adv_samples = np.concatenate(all_adv_samples, axis= 0)
    all_adv_labels = np.array(all_adv_labels)
    all_adv_attr = np.array(all_adv_attr)

    return all_adv_samples, all_adv_labels, all_adv_attr



def main(args):

    seed_set = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    all_acc = []
    all_fair = []

    for seed in seed_set:
        np.random.seed(seed)
        torch.manual_seed(seed)

        ## get train/test data
        X_train, X_test, X_valid, y_train, y_test, y_valid, a_train, a_test, a_valid = preprocess_adult_data(seed = args.seed)
        y_train = (y_train + 1) / 2
        y_test = (y_test + 1) / 2
        y_valid = (y_valid + 1) / 2

        #clf = LogisticRegression(fit_intercept=False).fit(X_train, y_train)    ## logistic regression
        #pred = clf.predict(X_test)
        #result4 = test_fairness(pred, y_test, a_test)
        #print(result4)

        ## initialize classifier optimizer
        model1 = classifier1(input_shape=X_train.shape[1]).cuda()
        optimizer1 = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        ## initialize adversary weight
        model2 = classifier2().cuda()
        optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        attack_num = int(X_train.shape[0] * args.ratio)

        ## generate poisonig samples

        if args.attack_type == 'mm':
            for epoch in range(10):
                adv_debias(model1, model2, X_train, y_train, a_train, optimizer1, optimizer2, 0.0)
        else:
            for epoch in range(10):
                adv_debias(model1, model2, X_train, y_train, a_train, optimizer1, optimizer2, args.alpha)

        #pred = evaluate(model1, X_test)
        #result4 = test_fairness(pred * 2 - 1, y_test * 2 - 1, a_test)
        #print('pretrained model performance', result4)

        if args.attack_type == 'label_flip':
            print('Doing Label Flipping Attack')
            attack_set = np.random.choice(X_train.shape[0], attack_num, replace=False)
            X_train1 = np.copy(X_train)
            y_train1 = np.copy(y_train)
            a_train1 = np.copy(a_train)
            y_train1[attack_set] = 1-y_train1[attack_set]
        elif args.attack_type == 'attr_flip':
            print('Doing Attr. Flipping Attack')
            attack_set = np.random.choice(X_train.shape[0], attack_num, replace=False)
            X_train1 = np.copy(X_train)
            y_train1 = np.copy(y_train)
            a_train1 = np.copy(a_train)
            a_train1[attack_set] = 1-a_train1[attack_set]
        elif args.attack_type == 'mm':
            X1 = np.mean(X_train[y_train == args.cls], axis=0)
            X_new, y_new = pgd_attack(model1, X1, X_train, y_train, attack_num, args.cls)
            X_train1 = np.concatenate([X_train, X_new], axis=0)
            y_train1 = np.concatenate([y_train, y_new])
            a_new = np.ones(attack_num) * args.cla
            a_train1 = np.concatenate([a_train, a_new])
        elif args.attack_type == 'fad':
            X1 = np.mean(X_train[y_train == args.cls], axis=0)
            X_new, y_new, a_new = pgd_attack_fair(model1, model2, X1, X_train, y_train, a_train, attack_num, args.alpha2, args.cls, args.cla)
            X_train1 = np.concatenate([X_train, X_new], axis=0)
            y_train1 = np.concatenate([y_train, y_new])
            a_train1 = np.concatenate([a_train, a_new])
        else:
            raise ValueError

        ## re-doing adv debiasing
        model11 = classifier1(input_shape=X_train.shape[1]).cuda()
        optimizer11 = optim.SGD(model11.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        model22 = classifier2().cuda()
        optimizer22 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        all_result = []
        all_valid_result = []

        for epoch in range(1, 31):
            #print('now epoch:  ' + str(epoch), flush=True)
            adv_debias(model11, model22, X_train1, y_train1, a_train1, optimizer11, optimizer22, args.alpha)
            pred = evaluate(model11, X_valid)
            result3 = test_fairness(pred * 2 - 1, y_valid * 2 - 1, a_valid)

            if args.alpha > 0:
                result3 = result3[0] - 3 * result3[1]
            else:
                result3 = result3[0] #- 3 * result3[1]

            all_valid_result.append(result3)
            pred = evaluate(model11, X_test)
            result4 = test_fairness(pred * 2 - 1, y_test * 2 - 1, a_test)
            all_result.append(result4)

        all_valid_result = np.array(all_valid_result)
        best_epoch = np.argmax(all_valid_result)
        best_test_result = np.array(all_result)[best_epoch]
        print(best_test_result, flush = True)

        all_acc.append(best_test_result[0])
        all_fair.append(1 - best_test_result[1])

    print('...............................')
    print('All Result')
    print(np.mean(np.array(all_acc)))
    print(np.mean(np.array(all_fair)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--alpha', type=float, default= 1.0)
    argparser.add_argument('--alpha2', type=float, default= 1.0)
    argparser.add_argument('--ratio', type=float, default= 0.5)
    argparser.add_argument('--cls', type=int, default= 0)
    argparser.add_argument('--cla', type=int, default= 0)
    argparser.add_argument('--attack_type', type=str, default= 'fad')
    args = argparser.parse_args()
    print(args)
    main(args)