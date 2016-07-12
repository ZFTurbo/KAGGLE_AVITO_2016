# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import xgboost as xgb
import random
from operator import itemgetter
import zipfile
from sklearn.metrics import roc_auc_score
import time
import shutil


random.seed(2016)


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


def run_train_with_model(train, features, model_path):
    start_time = time.time()

    gbm = xgb.Booster()
    gbm.load_model(model_path)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(train[features]))
    score = roc_auc_score(train['isDuplicate'].values, check)
    validation_df = pd.DataFrame({'itemID_1': train['itemID_1'].values, 'itemID_2': train['itemID_2'].values,
                                  'isDuplicate': train['isDuplicate'].values, 'probability': check})
    print('AUC score value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print('Prediction time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return validation_df, score


def create_submission(valid_prediction, score):
    now = datetime.datetime.now()
    sub_valid_file = './subm/submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '_train.csv'
    valid_prediction.to_csv(sub_valid_file, index=False)


def get_features(train):
    output = list(train.columns.values)
    output.remove('itemID_1')
    output.remove('itemID_2')
    output.remove('isDuplicate')
    return sorted(output)


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


def read_train_v2():
    print("Load train.csv")
    train = pd.read_csv("../modified_data/train_original.csv")
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

if 0:
    train, features = read_train_v2()
    excl = list_diff(list(train.columns.values), features)
    print('Length of train: ', len(train))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    print('Excluded [{}]: {}'.format(len(excl), sorted(excl)))
    train_prediction, score = run_train_with_model(train, features, '../run_0.94453/model_0.976687365593_eta_0.05_md_8_test_size_0.05.bin')
    create_submission(train_prediction, score)

    train, features = read_train()
    excl = list_diff(list(train.columns.values), features)
    print('Length of train: ', len(train))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    print('Excluded [{}]: {}'.format(len(excl), sorted(excl)))
    train_prediction, score = run_train_with_model(train, features, '../run_0.94292/model_0.977122196838_eta_0.04_md_9_test_size_0.02_iter_4971.bin')
    create_submission(train_prediction, score)

train, features = read_train()
excl = list_diff(list(train.columns.values), features)
print('Length of train: ', len(train))
print('Features [{}]: {}'.format(len(features), sorted(features)))
print('Excluded [{}]: {}'.format(len(excl), sorted(excl)))
train_prediction, score = run_train_with_model(train, features, '../run_0.94272/model_0.976438297669_eta_0.05_md_8_test_size_0.05_iter_4969.bin')
create_submission(train_prediction, score)
