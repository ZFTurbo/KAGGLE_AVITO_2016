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
import json

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
    importance = dict()
    create_feature_map(features)
    importance_arr = gbm.get_fscore(fmap='xgb.fmap')
    importance['default'] = sorted(importance_arr.items(), key=itemgetter(1), reverse=True)
    for f in ['weight', 'gain', 'cover']:
        try:
            importance_arr = gbm.get_score(fmap='xgb.fmap', importance_type=f)
            importance[f] = sorted(importance_arr.items(), key=itemgetter(1), reverse=True)
        except:
            importance[f] = 'Old version of XGBoost'
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def list_diff(a, b):
    return list(set(a) - set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def save_features_importance(out_file, features, imp):
    out = open(out_file, "w")
    out.write(str(features) + '\n\n')
    out.write(str(imp) + '\n\n')
    out.close()


def after_iteration(filename, n_iterations):
    def callback(env):
        if env.iteration > 0 and env.iteration % n_iterations == 0:
            env.model.save_model(filename)
            if 0:
                pd.Series(env.model.get_score(fmap=filename + '.fmap',
                                              importance_type='weight'))\
                                              .sort_values(ascending=False)\
                                              .to_csv(filename + '.weight.csv')
                pd.Series(env.model.get_score(fmap=filename + '.fmap',
                                              importance_type='gain'))\
                                              .sort_values(ascending=False)\
                                              .to_csv(filename + '.gain.csv')
                pd.Series(env.model.get_score(fmap=filename + '.fmap',
                                              importance_type='cover'))\
                                              .sort_values(ascending=False)\
                                              .to_csv(filename + '.cover.csv')
#        print(env.evaluation_result_list)
#        print(env.iteration % n_iterations)
#        print(env.model.attributes())
    return callback


# Save to JSON file
def store_json(data, path):
    f = open(path, 'w')
    json.dump(data, f)
    f.close()


# Read previously saved data from JSON file
def read_from_json(path):
    data = ''
    if os.path.isfile(path):
        f = open(path, 'r')
        data = json.load(f)
        f.close()
    else:
        print('No file {}'.format(path))
        exit()
    return data


def run_default_test(train, test, features, target, random_state=0):
    eta = 0.05
    max_depth = 8
    subsample = 0.9
    colsample_bytree = 0.8
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
        "seed": random_state
    }
    num_boost_round = 1000
    early_stopping_rounds = 100
    test_size = 0.05

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
    model_intermediate_file = os.path.join('models', 'temp_runtime_model.bin')
    model_iteration_num_file = os.path.join('models', 'last_iteration_num.txt')
    start_iteration = 0

    for run_number in range(20):
        if os.path.isfile(model_iteration_num_file):
            start_iteration = read_from_json(model_iteration_num_file)
            print('Start from iteration: {}'.format(start_iteration))
        if os.path.isfile(model_intermediate_file):
            print('Use saved model : {}'.format(model_intermediate_file))

        if 0:
            gbm = xgb.train(params, dtrain, num_boost_round,
                            evals=watchlist,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=True,
                            xgb_model=model_intermediate_file if os.path.isfile(model_intermediate_file) else None,
                            callbacks=[after_iteration(model_intermediate_file, 2)]
            )
        gbm = xgb.train(params, dtrain, num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True,
            xgb_model=model_intermediate_file if os.path.isfile(model_intermediate_file) else None,
        )
        print('Best iteration: {}'.format(gbm.best_iteration))

        print("Validating...")
        start_iteration = gbm.best_iteration
        check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
        score = roc_auc_score(X_valid[target].values, check)
        score_kaggle = auc(X_valid[target].values, check)
        print('Check error value: {:.6f} (Kaggle: {:.6f})'.format(score, score_kaggle))
        model_path = os.path.join('models', 'model_' + str(score) + '_eta_' + str(eta) + '_md_'
                                  + str(max_depth) + '_test_size_' + str(test_size)
                                  + '_iter_' + str(start_iteration) + '.bin')
        additional_data_path = os.path.join('models', 'model_' + str(score) + '_eta_' + str(eta) + '_md_'
                                            + str(max_depth) + '_test_size_' + str(test_size)
                                            + '_iter_' + str(start_iteration) + '_features_importance.txt')
        gbm.save_model(model_path)
        imp = get_importance(gbm, features)
        print('Importance array: ', imp)
        save_features_importance(additional_data_path, features, imp)
        gbm.save_model(model_intermediate_file)
        store_json(start_iteration, model_iteration_num_file)

        # Check model
        gbm1 = xgb.Booster()
        gbm1.load_model(model_path)
        check1 = gbm1.predict(xgb.DMatrix(X_valid[features]))
        score1 = roc_auc_score(X_valid[target].values, check1)
        print('Check model score: {}'.format(score1))

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def run_default_test_without_validation(train, features, target, random_state=0):
    eta = 0.03
    max_depth = 10
    subsample = 0.8
    colsample_bytree = 0.8
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
        "seed": random_state
    }
    num_boost_round = 1000

    print('Length train:', len(train.index))
    dtrain = xgb.DMatrix(train[features], train[target])

    watchlist = [(dtrain, 'train')]
    # gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, feval=auc_xgboost, verbose_eval=True)
    model_intermediate_file = os.path.join('models', 'temp_runtime_model.bin')
    model_iteration_num_file = os.path.join('models', 'last_iteration_num.txt')
    start_iteration = 0

    for run_number in range(5):
        if os.path.isfile(model_iteration_num_file):
            start_iteration = read_from_json(model_iteration_num_file)
            print('Start from iteration: {}'.format(start_iteration))
        if os.path.isfile(model_intermediate_file):
            print('Use saved model : {}'.format(model_intermediate_file))

        gbm = xgb.train(params, dtrain, num_boost_round,
            evals=watchlist,
            verbose_eval=True,
            xgb_model=model_intermediate_file if os.path.isfile(model_intermediate_file) else None,
        )
        print('Best iteration: {}'.format(gbm.best_iteration))

        print("Validating...")
        start_iteration = gbm.best_iteration
        score = 'unknown'
        model_path = os.path.join('models', 'model_' + str(score) + '_eta_' + str(eta) + '_md_'
                                  + str(max_depth) + '_iter_' + str(start_iteration) + '.bin')
        additional_data_path = os.path.join('models', 'model_' + str(score) + '_eta_' + str(eta) + '_md_'
                                            + str(max_depth) + '_iter_' + str(start_iteration) + '_features_importance.txt')
        gbm.save_model(model_path)
        imp = get_importance(gbm, features)
        print('Importance array: ', imp)
        save_features_importance(additional_data_path, features, imp)
        gbm.save_model(model_intermediate_file)
        store_json(start_iteration, model_iteration_num_file)


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist()


def run_test_with_model(train, test, features, target, random_state=0):
    start_time = time.time()
    test_size = 0.02

    # X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    split = round((1-test_size)*len(train.index))
    X_train = train[0:split]
    X_valid = train[split:]
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))

    # watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    # gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, feval=auc_xgboost, verbose_eval=True)
    # gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    gbm = xgb.Booster()
    gbm.load_model("models/model_0.968276662916_eta_0.2_md_5_test_size_0.02.bin")

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]))
    score = roc_auc_score(X_valid[target].values, check)
    score_kaggle = auc(X_valid[target].values, check)
    print('Check error value: {:.6f} (Kaggle: {:.6f})'.format(score, score_kaggle))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]))

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


def get_features(train):
    output = list(train.columns.values)
    output.remove('itemID_1')
    output.remove('itemID_2')
    output.remove('isDuplicate')
    return sorted(output)


def decrease_size_dataframe(df):
    float_decr = 0
    int8_decr = 0
    int16_decr = 0
    int32_decr = 0
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
            float_decr += 1
        elif df[col].dtype == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype(np.int8)
                int8_decr += 1
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype(np.int16)
                int16_decr += 1
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
                int32_decr += 1
    # df.to_hdf(f_store, key, format='t', complevel=9, complib='blosc')
    print('Float decrease: {}'.format(float_decr))
    print('Int8 decrease: {}'.format(int8_decr))
    print('Int16 decrease: {}'.format(int16_decr))
    print('Int32 decrease: {}'.format(int32_decr))


def read_test_train_old():
    print("Load train.csv")
    # train = pd.read_csv("../modified_data/train.csv")
    # train = pd.read_csv("../modified_data/train_ad_pairs.csv")
    train = pd.read_csv("../modified_data/train_original.csv")
    decrease_size_dataframe(train)
    # for f in list(train.columns.values):
    #    print(train[f].describe())
    # train = train.drop(['ids_equality'], axis=1)
    # print(list(train.columns.values)[280])
    # print(train[list(train.columns.values)[280]].describe())
    null_count = train.isnull().sum().sum()
    if null_count > 0:
        print('Nans:', null_count)
        cols = train.isnull().any(axis=0)
        print(cols[cols == True])
        rows = train.isnull().any(axis=1)
        print(rows[rows == True])
        print('NANs in train, please check it!')
        exit()
    print("Load test.csv")
    test = pd.read_csv("../modified_data/test.csv")
    decrease_size_dataframe(test)
    null_count = test.isnull().sum().sum()
    if null_count > 0:
        print('Nans:', null_count)
        cols = test.isnull().any(axis=0)
        print(cols[cols == True])
        print('NANs in test, please check it!')
        exit()
    # test.fillna(-1, inplace=True)
    features = get_features(train, test)
    return train, test, features


def read_train():
    print("Load train.csv")
    train = pd.read_hdf("../modified_data/train_original.csv.hdf", 'table')
    null_count = train.isnull().sum().sum()
    if null_count > 0:
        print('Nans:', null_count)
        cols = train.isnull().any(axis=0)
        print(cols[cols == True])
        rows = train.isnull().any(axis=1)
        print(rows[rows == True])
        print('NANs in train, please check it!')
        exit()
    features = get_features(train)
    return train, features


train, features = read_train()
excl = list_diff(list(train.columns.values), features)
print('Length of train: ', len(train))
print('Features [{}]: {}'.format(len(features), sorted(features)))
print('Excluded [{}]: {}'.format(len(excl), sorted(excl)))
# test_prediction, score = run_default_test(train, test, features, 'isDuplicate')
test_prediction, score = run_default_test_without_validation(train, features, 'isDuplicate')
# test_prediction, score = run_test_with_model(train, test, features, 'isDuplicate')
# print('Real score = {}'.format(score))
# create_submission(score, test, test_prediction)


# LS: 0.932189 LB: 0.84365 - Long run
# LS: 0.923190 LB: 0.84134 - Train only on last part of data
# LS: 0.895049 LB: 0.83615 - epochs = 20, eta = 0.2, max_depth = 10 0.5 from train
# LS: 0.918594 LB: 0.83324 - epochs = 20, depth = 20
# LS: 0.912189 LB: 0.84492 - epochs = 5000, depth = 3
# LS: 0.934948 LB: 0.91291 - epochs = 100, depth = 3, eta = 0.2
# LS: 0.955502 LB: 0.92396 - epochs = 5000, depth = 3, eta = 0.2, test_size = 0.1
# LS: 0.954937 LB: 0.92376 - epochs = 5000 (2683), depth = 3, eta = 0.2, test_size = 0.05
# LS: 0.955618 LB: 0.92393 - epochs = 5000 (3451), depth = 3, eta = 0.2, test_size = 0.02
# LS: 0.958303 LB: 0.92495 - epochs = 5000 (2599), depth = 4, eta = 0.2, test_size = 0.02
# LS: 0.958776 LB: 0.92527 - epochs = 5000 (1191), depth = 5, eta = 0.2, test_size = 0.02
# LS: 0.960701 LB: 0.92620 - epochs = 5000 (3572), depth = 5, eta = 0.1, test_size = 0.02
# LS: 0.960472 LB: 0.92659 - epochs = 5000 (3477), depth = 5, eta = 0.1, test_size = 0.02 (Modified train)
# LS: 0.961548 LB: 0.92692 - epochs = 5000 (2245), depth = 6, eta = 0.1, test_size = 0.02 (Modified train)
# LS: 0.962546 LB: 0.92684 - epochs = 5000 (1630), depth = 7, eta = 0.1, test_size = 0.02 (Modified train) - slightly worse
# LS: 0.964071 LB: 0.92751 - epochs = 10000 (9180), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.7 - CS: 0.7)
# LS: 0.964182 LB: 0.92748 - epochs = 10000 (9751), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.8 - CS: 0.8) - slightly worse
# LS: 0.963186 LB: 0.92717 - epochs = 10000 (7945), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.6 - CS: 0.6)
# LS: 0.963882 LB: 0.92774 - epochs = 10000 (8922), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Damerau-Levin
# LS: 0.965184 LB: 0.93051 - epochs = 15000 (8922), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (570.94 minutes)
# LS: 0.966756 LB: 0.93251 - epochs = 15000 (7474), depth = 6, eta = 0.05, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (1582.71 minutes)
# LS: 0.968624 LB: 0.93287 - epochs = 15000 (9099), depth = 7, eta = 0.04, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (2334 minutes)
# LS: 0.969687 LB: 0.93294 - epochs = 15000 (8415), depth = 8, eta = 0.04, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) (2806 minutes)
# LS: 0.965264 LB: 0.93130 - epochs = 15000 (3125), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Many JSON Params (872.94 minutes)
# LS: 0.963257 LB: 0.92629 - epochs = 15000 (2111), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.7 - CS: 0.7) - Use only last 0.5 of train data (384 minutes) - BAD!
# LS: 0.965393 LB: 0.92582 - epochs = 15000 (5681), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.8 - SS: 0.9) - Removed strange pairs from train (722.94 minutes) - BAD!
# LS: 0.970019 LB: 0.92843 - epochs = 15000 (2860), depth = 8, eta = 0.2, test_size = 0.02 (CSBT: 0.8 - SS: 0.9) - Default input (1333 minutes)
# LS: 0.966588 LB: 0.93135 - epochs = 15000 (4883), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.8 - SS: 0.9) - Extended input (3117753) (1548 minutes)
# LS: 0.968245 LB: 0.93546 - epochs = 15000 (4000), depth = 5, eta = 0.2, test_size = 0.02 (CSBT: 0.8 - SS: 0.9) - Default input, new params (???? minutes)
# LS: 0.976678 LB: 0.94453 - epochs = 20000 (10000), depth = 8, eta = 0.05, test_size = 0.05 (CSBT: 0.7 - SS: 0.7) - Default input, 512 Features (5972 minutes)
# LS: 0.976437 LB: 0.94272 - epochs = 20000 (4969), depth = 8, eta = 0.05, test_size = 0.05 (CSBT: 0.8 - SS: 0.9) - Default input, 559 Features
# LS: 0.977122 LB: 0.94292 - epochs = 20000 (4971), depth = 9, eta = 0.04, test_size = 0.02 (CSBT: 0.7 - SS: 0.7) - Default input, 559 Features
# LS: 0.976655 LB: 0.94270 - epochs = 20000 (5971), depth = 8, eta = 0.05, test_size = 0.05 (CSBT: 0.8 - SS: 0.9) - Default input, 559 Features
# LS: 0.977265 LB: 0.94284 - epochs = 20000 (5873), depth = 9, eta = 0.04, test_size = 0.02 (CSBT: 0.7 - SS: 0.7) - Default input, 559 Features