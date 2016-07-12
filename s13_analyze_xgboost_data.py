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
import math

random.seed(2016)


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
    train = pd.read_csv("../modified_data/train.csv")
    # print(train['categoryID_same'].describe())
    # print(train['categoryID_same'].unique())
    # exit()
    # train = pd.read_csv("../modified_data/train_ad_pairs.csv")
    train.fillna(-1, inplace=True)
    print("Load test.csv")
    test = pd.read_csv("../modified_data/test.csv")
    test.fillna(-1, inplace=True)
    features = get_features(train, test)
    return train, test, features


def append_items_info(train, items):
    items = items.rename(
        columns={
            'itemID': 'itemID_1',
        }
    )
    train = pd.merge(train, items, how='left', on='itemID_1', left_index=True)
    items = items.rename(
        columns={
            'itemID_1': 'itemID_2',
        }
    )
    train = pd.merge(train, items, how='left', on='itemID_2', left_index=True)
    items = items.rename(
        columns={
            'itemID_2': 'itemID',
        }
    )
    return train


def print_debug_data(out, table, features, loc, pred, real):
    out.write("Predicted value: {}<BR>\n".format(pred))
    out.write("Real value: {}<BR>\n".format(real))
    for f in list(features):
        out.write("{} {}<BR>\n".format(f, table[f].iloc[loc]))

    out.write('<table border=1><tr><td></td><td>ITEM 1</td><td>ITEM 2</td></tr>\n\n')

    out.write('<tr>')
    out.write('<td>ID</td>')
    out.write('<td>' + str(table['itemID_1'].iloc[loc]) + '</td>')
    out.write('<td>' + str(table['itemID_2'].iloc[loc]) + '</td>')
    out.write('</tr>')

    for f in ['categoryID', 'locationID', 'metroID', 'lat', 'lon', 'price', 'attrsJSON', 'title', 'description']:
        out.write('<tr>\n')
        out.write('<td>' + f + '</td>\n')
        out.write('<td>' + str(table[f + '_x'].iloc[loc]) + '</td>\n')
        out.write('<td>' + str(table[f + '_y'].iloc[loc]) + '</td>\n')
        out.write('</tr>\n')

    out.write('<tr>')
    out.write('<td>Images</td>')
    for f in ['images_array_x', 'images_array_y']:
        out.write('<td>')
        arr = str(table[f].iloc[loc]).split(',')
        for i in range(len(arr)):
            im_id = int(arr[i])
            folder1 = int(math.floor(im_id % 100)/10)
            folder2 = int(im_id % 100)
            im_path = "../../input/Images_" + str(folder1) + "/" + str(folder2) + "/" + str(im_id) + ".jpg"
            out.write(str(im_id) + '<BR>\n')
            out.write(" <img style=\"min-width: 300px;\" src=\"" + im_path + "\"> <BR>\n")
        out.write('</td>')
    out.write('</tr>')

    out.write('</table>\n\n')
    out.write('<BR><BR>\n\n')


def output_critical_tests(train, features, target, model_path, test_size):
    out_path = "cache/fails.html"
    out = open(out_path, "w", encoding='utf-8')
    gbm = xgb.Booster()
    gbm.load_model(model_path)

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)
    items.fillna(-1, inplace=True)

    split = round((1-test_size)*len(train.index))
    X_train = train[0:split]
    X_valid = train[split:]
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]))
    # print(X_valid[features][:100])
    # print(check[:100])
    score = roc_auc_score(X_valid[target].values, check)
    print('Score: {}'.format(score))

    X_valid = append_items_info(X_valid, items)

    count = 0
    for i in range(len(X_valid[target].values)):
        if abs(X_valid[target].values[i] - check[i]) > 0.9:
            print(X_valid[target].values[i], check[i])
            if count > 100:
                break
            print_debug_data(out, X_valid, features, i, check[i], X_valid[target].values[i])
            count += 1
    print('Count critical: {} from {}'.format(count, len(check)))
    out.close()


train, test, features = read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
output_critical_tests(train, features, 'isDuplicate', 'models/model_0.960701703647_eta_0.1_md_5_test_size_0.02.bin', 0.02)

