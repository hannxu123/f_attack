
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD

def get_adult_data():
    '''
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
    '''

    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']
    train = pd.read_csv('./dataset/adult.data', header = None)
    test = pd.read_csv('./dataset/adult.test', header = None)

    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers
    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    ## remove missing values
    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)

    ## one-hot method to preprocess categorical data
    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-stataus',
               'occupation', 'relationship', 'race', 'sex', 'native-country'])

    ## remove rare
    #delete_these = ['race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other', 'sex_ Female']
    delete_these = ['sex_ Female']
    delete_these += ['native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia',
                     'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador',
                     'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala',
                     'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong',
                     'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland',
                     'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos',
                     'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru',
                     'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico',
                     'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand',
                     'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']
    delete_these += ['fnlwgt']
    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_ Male'])


def preprocess_adult_data(seed = 0):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)

    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset and split into train and test
    dataset_orig = get_adult_data()

    # we will standardize continous features
    #continous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    #continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]
    #dataset_orig_train, dataset_orig_test = dataset_orig.split([0.75], shuffle=True, seed = seed)

    #SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    #dataset_orig_train.features[:, continous_features_indices] = SS.transform(dataset_orig_train.features[:, continous_features_indices])
    #dataset_orig_test.features[:, continous_features_indices] = SS.transform(dataset_orig_test.features[:, continous_features_indices])

    X = dataset_orig.features
    y = dataset_orig.labels

    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y.reshape(-1,1))
    y = one_hot.transform(y.reshape(-1,1))
    y = (y[:, 1] - 0.5) * 2

    # Also create a train/test set where the predictive features (X) do not include gender and gender is what you want to predict (y).
    X = np.delete(X, [dataset_orig.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)
    a = dataset_orig.features[:, dataset_orig.feature_names.index('sex_ Male')]
    one_hot.fit(a.reshape(-1,1))
    a = one_hot.transform(a.reshape(-1,1))[:,1]

    ## pca
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    neg_idx = np.random.choice(neg_idx, pos_idx.shape[0], replace=False)
    X = np.concatenate([X[pos_idx], X[neg_idx]], axis=0)
    y = np.concatenate([y[pos_idx], y[neg_idx]], axis=0)
    a = np.concatenate([a[pos_idx], a[neg_idx]], axis=0)

    ## doing pca
    mean = np.mean(X, axis=0)
    X = X - mean
    svd = TruncatedSVD(n_components=15)
    svd.fit(X[:,5:])
    X2 = svd.transform(X[:,5:])
    X = np.concatenate([X[:,0:5], X2], axis= 1)
    SS = StandardScaler().fit(X)
    X = SS.transform(X)

    ## split
    random_idx = np.random.uniform(0, 1, X.shape[0])
    train_idx = random_idx < 0.6
    test_idx = (random_idx >= 0.6) & (random_idx < 0.8)
    valid_idx = (random_idx >= 0.8)
    X_train = X[train_idx]
    y_train = y[train_idx]
    a_train = a[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]
    a_test = a[test_idx]

    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    a_valid = a[valid_idx]

    return X_train, X_test, X_valid, y_train, y_test, y_valid, a_train, a_test, a_valid

