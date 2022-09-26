import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp


################## loss functions

def log_logistic_torch(X):
    out = torch.empty_like(X)  # same dimensions and data types
    idx = X > 0
    out[idx] = -torch.log(1.0 + torch.exp(-X[idx]))
    out[~idx] = X[~idx] - torch.log(1.0 + torch.exp(X[~idx]))
    return out

def logistic_loss_torch(w, X, y):
    yz = y * torch.matmul(w, X)
    out = - (log_logistic_torch(yz))
    return out

def lagrangian2(w, x, y, a, nu, clean_num):

    ## output vector
    loss_vec = logistic_loss_torch(w, x.T, y)

    FNR_0 = torch.sum((a[0:clean_num] == 0) * (y[0:clean_num]  == 1) * loss_vec[0:clean_num] ) \
            / float(torch.sum((a[0:clean_num]  == 0) * (y[0:clean_num]  == 1)))
    FNR_1 = torch.sum((a[0:clean_num] == 1) * (y[0:clean_num]  == 1) * loss_vec[0:clean_num] ) \
            / float(torch.sum((a[0:clean_num]  == 1) * (y[0:clean_num]  == 1)))

    c0 = torch.relu(torch.abs(FNR_0 - FNR_1) - nu)
    lag = torch.mean(loss_vec) + c0 * 3
    return lag, c0.item()


#################### inner maximization
def constraint_attack(w, y, d, center, bar):

    x0 = cp.Variable((d,1))

    objective = cp.Minimize(y * w @ x0)   ## maximize inner loss

    all_constraint = []
    all_constraint += [cp.sum_squares(x0 - center.reshape(x0.shape[0], 1)) <= bar]

    prob = cp.Problem(objective, all_constraint)
    prob.solve()

    score = prob.value
    return x0.value.T, score


def cert_fair_attack(x, y, x_control, weight, number, aa, bar):

    # set tensor
    weight = torch.tensor(weight).float().cuda()

    all_x = []
    all_y = []
    all_a = []
    clean_num = x.shape[0]

    x = torch.tensor(x, dtype=torch.float).cuda()
    y = torch.tensor(y).cuda()
    a = torch.tensor(x_control).cuda()

    nu = 0.03
    y_list = [-1.0, 1.0]
    d = x.shape[1]

    for i in range(int(number * 2)):

        score_max= -1000
        for yy in (y_list):

            center = torch.mean(x[y == yy], dim=0)
            xx, score = constraint_attack(weight.detach().cpu().numpy(), yy, d, center.detach().cpu().numpy(), bar)
            if score > score_max:
                x_new = xx
                y_new = yy
                a_new = aa

        all_x.append(x_new)
        all_y.append(y_new)
        all_a.append(a_new)

        # concatenate to get the new dataset with a sampled subset of total set
        x_new = torch.tensor(x_new, dtype = torch.float).cuda().repeat(number , 1)
        y_new = torch.ones((number)).cuda() * y_new
        a_new = torch.ones((number)).cuda() * a_new

        x_total = torch.cat([x, x_new], dim = 0)
        y_total = torch.cat([y, y_new])
        a_total = torch.cat([a, a_new])

        # do one step gradient descent to minimize the outside
        weight.requires_grad_()
        out_loss, c0 = lagrangian2(weight, x_total, y_total, a_total, nu, clean_num)
        weight_grad = torch.autograd.grad(out_loss, weight)[0]
        weight = weight - weight_grad * 0.1
        weight.requires_grad_()
        weight.retain_grad()

        ## constrain the space of model w
        if torch.norm(weight) > 5:
            weight = weight / torch.norm(weight) * 5

        #if (i % 100 == 0):
        #    print('In Loss ', score, 'Out Loss ', out_loss.item(), 'Unfair ', c0, flush= True)

    all_x = np.concatenate(all_x, axis= 0)[number:]
    all_y = np.array(all_y)[number:]
    all_a = np.array(all_a)[number:]
    all_x = all_x + np.random.normal(0, 0.05, all_x.shape)

    return all_x, all_y, all_a
