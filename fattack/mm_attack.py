import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp


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


def constraint_attack(w, y, d, center, bar):

    x0 = cp.Variable((d,1))

    objective = cp.Minimize( y * w @ x0)   ## maximize inner loss

    all_constraint = []
    all_constraint += [cp.sum_squares(x0 - center.reshape(x0.shape[0], 1)) <= bar]

    prob = cp.Problem(objective, all_constraint)
    prob.solve()

    score = prob.value
    return x0.value.T, score



def cert_attack(x, y, weight, number, bar):

    # base categorical features
    d = x.shape[1]

    # set tensor
    x = torch.tensor(x).float().cuda()
    weight = torch.tensor(weight).float().cuda()
    y = torch.tensor(y).float().cuda()

    all_x = []
    all_y = []
    y_list = [-1.0, 1.0]

    for i in range(number * 2):

        score_max= -1000
        for yy in (y_list):
            center = torch.mean(x[y == yy], dim=0)
            xx, score = constraint_attack(weight.detach().cpu().numpy(), yy, d, center.detach().cpu().numpy(), bar)
            if score > score_max:
                x_new = xx
                y_new = yy

        # record the generated attacks
        all_x.append(x_new)
        all_y.append(y_new)

        # concatenate to get the new dataset
        x_new = torch.tensor(x_new, dtype = torch.float).cuda()
        y_new = torch.tensor(y_new, dtype = torch.float).cuda()

        x_new = x_new.repeat(number, 1)
        y_new = y_new.repeat(number)
        x_total = torch.cat([x, x_new.cuda()], axis=0)
        y_total = torch.cat([y, y_new.cuda()], axis=0)

        # do one step gradient descent to minimize the outside
        weight.requires_grad_()
        out_loss = logistic_loss_torch(weight, x_total.T, y_total)
        weight_grad = torch.autograd.grad(out_loss, weight)[0]
        weight = weight - weight_grad

        weight.detach()
        weight.requires_grad_()
        weight.retain_grad()

    all_x = np.concatenate(all_x, axis = 0)[number:]
    all_y = np.array(all_y)[number:]
    all_x = all_x + np.random.normal(0, 0.05, all_x.shape)

    return all_x, all_y



