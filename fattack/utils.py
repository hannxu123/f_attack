
import numpy as np
import matplotlib.pyplot as plt

def compute_balanced_accuracy(pred, true_label):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = np.sum((true_label == 1) & (pred == 1)) / np.sum(true_label == 1)
    TNR = np.sum((true_label != 1) & (pred != 1)) / np.sum(true_label != 1)

    print('test accuracy ' + str(np.sum(pred == true_label) / true_label.shape[0]))
    print('balanced accuracy ' +str(0.5 * (TPR+TNR)))
    print('TPR ' +str((TPR)) + ' TNR ' +str(TNR))
    print('...........................')



def test_fairness(pred, y_test, attr):

    TPR_prot = np.sum((attr == 0) * (y_test == 1) * (pred == 1)) / float(np.sum((attr == 0) * (y_test == 1)))
    TPR_unprot = np.sum((attr == 1) * (y_test == 1) * (pred == 1)) / float(np.sum((attr == 1) * (y_test == 1)))

    #acc = np.round(np.sum(y_test == pred) / y_test.shape[0], 3)
    diff_odd1 = np.round(abs(TPR_prot - TPR_unprot), 5)

    ## balanced accuracy
    acc_pos = np.sum((y_test == 1) & (pred == 1)) / float(np.sum((y_test == 1)))
    acc_neg = np.sum((y_test != 1) & (pred != 1)) / float(np.sum((y_test != 1)))
    balanced_acc = np.round((acc_pos  +  acc_neg ) / 2, 5)

    return balanced_acc, diff_odd1


