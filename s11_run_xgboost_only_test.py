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


def run_test_with_model(train, test, features, model_path):
    start_time = time.time()

    gbm = xgb.Booster()
    gbm.load_model(model_path)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(train[features]))
    score = roc_auc_score(train['isDuplicate'].values, check)
    validation_df = pd.DataFrame({'isDuplicate': train['isDuplicate'].values, 'probability': check})
    # print(validation_df)
    print('AUC score value: {:.6f}'.format(score))
    # score1 = roc_auc_score(validation_df['isDuplicate'].values, validation_df['probability'])
    # print('AUC score check value: {:.6f}'.format(score1))


    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]))

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), validation_df, score


def create_submission(test, prediction, valid_prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = './subm/submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    sub_valid_file = './subm/submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '_validation.csv'
    valid_prediction.to_csv(sub_valid_file, index=False)
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
    output = intersect(trainval, testval)
    output.remove('itemID_1')
    output.remove('itemID_2')
    return sorted(output)


def read_test_train(train_size):
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
    split = round((1-train_size)*len(train.index))
    train = train[split:]
    print("Load test.csv")
    test = pd.read_hdf("../modified_data/test.hdf", 'table')
    null_count = test.isnull().sum().sum()
    if null_count > 0:
        print('Nans:', null_count)
        cols = test.isnull().any(axis=0)
        print(cols[cols == True])
        print('NANs in test, please check it!')
        exit()
    features = get_features(train, test)
    return train, test, features


def read_test_train_v2(train_size):
    print("Load train.csv")
    train = pd.read_csv("../modified_data/train_original.csv")
    split = round((1-train_size)*len(train.index))
    train = train[split:]
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
    null_count = test.isnull().sum().sum()
    if null_count > 0:
        print('Nans:', null_count)
        cols = test.isnull().any(axis=0)
        print(cols[cols == True])
        print('NANs in test, please check it!')
        exit()
    features = get_features(train, test)
    return train, test, features


test_size = 0.02
train, test, features = read_test_train(test_size)
excl = list_diff(list(train.columns.values), features)
print('Length of test: ', len(test))
print('Length of train: ', len(train))
print('Features [{}]: {}'.format(len(features), sorted(features)))
print('Excluded [{}]: {}'.format(len(excl), sorted(excl)))
# test_prediction, valid_prediction, score = run_test_with_model(train, test, features, '../run_0.94292/model_0.977122196838_eta_0.04_md_9_test_size_0.02_iter_4971.bin')
# test_prediction, valid_prediction, score = run_test_with_model(train, test, features, '../run_0.94272/model_0.976438297669_eta_0.05_md_8_test_size_0.05_iter_4969.bin')
# test_prediction, valid_prediction, score = run_test_with_model(train, test, features, '../run_0.94453/model_0.976687365593_eta_0.05_md_8_test_size_0.05.bin')
# test_prediction, valid_prediction, score = run_test_with_model(train, test, features, '../run_0.94xv4/model_unknown_eta_0.03_md_10_iter_2999.bin')
test_prediction, valid_prediction, score = run_test_with_model(train, test, features, '../run_0.94xv5/model_unknown_eta_0.02_md_11_iter_3999.bin')
create_submission(test, test_prediction, valid_prediction)
