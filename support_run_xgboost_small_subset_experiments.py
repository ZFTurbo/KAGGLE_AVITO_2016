# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
import statistics
import pickle
import os
from heapq import nlargest
from operator import itemgetter
import zipfile
from sklearn.metrics import roc_auc_score
import time
import shutil

random.seed(2016)


def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r


def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x == 1])
    num_negative = len(actual) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i] == 1])
    auc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) /
           (num_negative * num_positive))
    return auc


def auc_xgboost(preds, real):
    return "auc", auc(real.get_label(), preds)


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def run_default_test(train, test, features, target, config, random_state=0):
    eta = config['eta']
    max_depth = config['max_depth']
    subsample = config['subsample']
    colsample_bytree = config['colsample_bytree']
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
        "nthreads": 6,
    }
    num_boost_round = 30000
    early_stopping_rounds = 150
    test_size = config['test_size']

    # X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    split = round((1-test_size)*len(train.index))
    X_train = train[0:split]
    X_valid = train[split:]
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    # gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, feval=auc_xgboost, verbose_eval=True)
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = roc_auc_score(X_valid[target].values, check)
    score_kaggle = auc(X_valid[target].values, check)
    print('Check error value: {:.6f} (Kaggle: {:.6f})'.format(score, score_kaggle))
    model_path = os.path.join('models', 'model_' + str(score) + '_eta_' + str(eta) + '_md_' + str(max_depth) + '_test_size_' + str(test_size) + '.bin')
    gbm.save_model(model_path)

    # Check model
    gbm1 = xgb.Booster()
    gbm1.load_model(model_path)
    check1 = gbm1.predict(xgb.DMatrix(X_valid[features]))
    score1 = roc_auc_score(X_valid[target].values, check1)
    print('Check model score: {}'.format(score1))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = './subm/submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('id,probability\n')
    total = 0
    for id in test['id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

    print('Creating zip-file...')
    z = zipfile.ZipFile(sub_file + ".zip", "w", zipfile.ZIP_DEFLATED)
    z.write(sub_file)
    z.close()

    # Copy code
    shutil.copy2(__file__, sub_file + ".py")


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    # print(trainval)
    # print(testval)
    output = intersect(trainval, testval)
    output.remove('itemID_1')
    output.remove('itemID_2')
    return sorted(output)


def read_test_train():
    print("Load train.csv")
    # train = pd.read_csv("../modified_data/train.csv")
    train = pd.read_csv("../modified_data/train_ad_pairs_subpart.csv")
    print('Nans:', train.isnull().sum().sum())
    # train.fillna(-1, inplace=True)
    print("Load test.csv")
    test = pd.read_csv("../modified_data/test.csv")
    print('Nans:', test.isnull().sum().sum())
    # test.fillna(-1, inplace=True)
    features = get_features(train, test)
    return train, test, features


def gen_subset():
    print("Load train.csv")
    start = 2000000
    train = pd.read_csv("../modified_data/train_original.csv")[start:]
    train.to_csv("../modified_data/train_ad_pairs_subpart.csv", index=False)
    exit()

# gen_subset()
train, test, features = read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))

params = []
params.append({"eta": 0.02, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.8, "test_size": 0.04})


for i in range(len(params)):
    out = open('log.txt', "a")
    test_prediction, score = run_default_test(train, test, features, 'isDuplicate', params[i])
    print('Params = {}'.format(params[i]))
    print('Real score = {}'.format(score))
    out.write('Params = {}'.format(params[i]))
    out.write('Real score = {}\n'.format(score))
    create_submission(score, test, test_prediction)
    out.close()

# LS: 0.963882 LB: 0.92774 - epochs = 10000 (8922), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Damerau-Levin
# LS: 0.965184 LB: 0.93051 - epochs = 15000 (8922), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (570.94 minutes)
# LS: 0.966756 LB: 0.93251 - epochs = 15000 (7474), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (1582.71 minutes)
# LS: 0.968624 LB: 0.93287 - epochs = 15000 (9099), depth = 7, eta = 0.04, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (2334 minutes)
# LS: 0.969687 LB: 0.93294 - epochs = 15000 (8415), depth = 8, eta = 0.04, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) (2806 minutes)
# LS: 0.965264 LB: 0.93130 - epochs = 15000 (3125), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (872.94 minutes)
# LS: 0.963257 LB: 0.92629 - epochs = 15000 (2111), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Use only last 0.5 of train data (384 minutes) - BAD!

# Experiments with small subset (10%)
# LS:  LB: - epochs = 15000 (2111), depth = 5, eta = 0.2, test_size = 0.04 (CSBT: 0.7 - CS: 0.7)
'''
Params = {'max_depth': 5, 'eta': 0.2, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9539845972476604
Params = {'max_depth': 5, 'eta': 0.1, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9567561705611016
Params = {'max_depth': 5, 'eta': 0.05, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.957901713272199
Params = {'max_depth': 5, 'eta': 0.02, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9579410498561073
Params = {'max_depth': 6, 'eta': 0.02, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9595064197304938
Params = {'max_depth': 7, 'eta': 0.02, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9595416691867927
Params = {'max_depth': 8, 'eta': 0.02, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9600238533167704
Params = {'max_depth': 9, 'eta': 0.02, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9609354120108953
Params = {'max_depth': 10, 'eta': 0.02, 'colsample_bytree': 0.7, 'test_size': 0.04, 'subsample': 0.7}Real score = 0.9599087352482156
Params = {'test_size': 0.04, 'colsample_bytree': 0.5, 'eta': 0.02, 'max_depth': 6, 'subsample': 0.5}Real score = 0.9582514615439094
Params = {'test_size': 0.04, 'colsample_bytree': 0.6, 'eta': 0.02, 'max_depth': 6, 'subsample': 0.5}Real score = 0.9586712079339258
Params = {'test_size': 0.04, 'colsample_bytree': 0.7, 'eta': 0.02, 'max_depth': 6, 'subsample': 0.5}Real score = 0.9581280399828993
Params = {'test_size': 0.04, 'colsample_bytree': 0.8, 'eta': 0.02, 'max_depth': 6, 'subsample': 0.5}Real score = 0.9588052107936873
Params = {'test_size': 0.04, 'colsample_bytree': 0.9, 'eta': 0.02, 'max_depth': 6, 'subsample': 0.5}Real score = 0.9594487476136344
Params = {'test_size': 0.04, 'colsample_bytree': 0.5, 'eta': 0.02, 'max_depth': 6, 'subsample': 0.6}Real score = 0.9583685827896322
Params = {'eta': 0.02, 'subsample': 0.6, 'max_depth': 6, 'colsample_bytree': 0.6, 'test_size': 0.04}Real score = 0.9594360177458233
Params = {'eta': 0.02, 'subsample': 0.6, 'max_depth': 6, 'colsample_bytree': 0.7, 'test_size': 0.04}Real score = 0.9593208027493414
Params = {'eta': 0.02, 'subsample': 0.6, 'max_depth': 6, 'colsample_bytree': 0.8, 'test_size': 0.04}Real score = 0.9592006444285567
Params = {'eta': 0.02, 'subsample': 0.6, 'max_depth': 6, 'colsample_bytree': 0.9, 'test_size': 0.04}Real score = 0.9596419088184769
Params = {'eta': 0.02, 'subsample': 0.7, 'max_depth': 6, 'colsample_bytree': 0.5, 'test_size': 0.04}Real score = 0.9596003267375818
Params = {'eta': 0.02, 'subsample': 0.7, 'max_depth': 6, 'colsample_bytree': 0.6, 'test_size': 0.04}Real score = 0.9596211823966477
Params = {'eta': 0.02, 'subsample': 0.7, 'max_depth': 6, 'colsample_bytree': 0.7, 'test_size': 0.04}Real score = 0.9600039669369792
Params = {'eta': 0.02, 'subsample': 0.7, 'max_depth': 6, 'colsample_bytree': 0.8, 'test_size': 0.04}Real score = 0.9596103587780773
Params = {'colsample_bytree': 0.9, 'subsample': 0.7, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.9600754997474704
Params = {'colsample_bytree': 0.5, 'subsample': 0.8, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.959891385149194
Params = {'colsample_bytree': 0.6, 'subsample': 0.8, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.9584947183326121
Params = {'colsample_bytree': 0.7, 'subsample': 0.8, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.9600271650209597
Params = {'colsample_bytree': 0.8, 'subsample': 0.8, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.9603737954443357
Params = {'colsample_bytree': 0.9, 'subsample': 0.8, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.9601874353490725
Params = {'colsample_bytree': 0.5, 'subsample': 0.9, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.9600879226768444
Params = {'colsample_bytree': 0.6, 'subsample': 0.9, 'test_size': 0.04, 'max_depth': 6, 'eta': 0.02}Real score = 0.959957005356108
Params = {'max_depth': 6, 'colsample_bytree': 0.7, 'test_size': 0.04, 'eta': 0.02, 'subsample': 0.9}Real score = 0.9591044273058636
Params = {'max_depth': 6, 'colsample_bytree': 0.8, 'test_size': 0.04, 'eta': 0.02, 'subsample': 0.9}Real score = 0.9603775433241989
Params = {'max_depth': 6, 'colsample_bytree': 0.9, 'test_size': 0.04, 'eta': 0.02, 'subsample': 0.9}Real score = 0.9595172433490641
Params = {'max_depth': 6, 'colsample_bytree': 0.95, 'test_size': 0.04, 'eta': 0.02, 'subsample': 0.95}Real score = 0.9599978766322016
Params = {'max_depth': 6, 'colsample_bytree': 0.99, 'test_size': 0.04, 'eta': 0.02, 'subsample': 0.99}Real score = 0.9595161932965162
Params = {'subsample': 0.9, 'colsample_bytree': 0.8, 'eta': 0.02, 'max_depth': 6, 'test_size': 0.04}Real score = 0.9603775433241989
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.02, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9611709175394486
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.06, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9606459668911368
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.01, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9561412147596019
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.08, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.960773268065593
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.09, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9606510021357997
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.1, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9600772921956262
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.2, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9595230981915533
Params = {'eta': 0.02, 'subsample': 0.9, 'test_size': 0.5, 'max_depth': 6, 'colsample_bytree': 0.8}Real score = 0.9559741947482633
'''