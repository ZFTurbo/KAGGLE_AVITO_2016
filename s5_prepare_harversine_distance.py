# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
import time
import math
from haversine import haversine


def distance_haversine(row):
    if row['lat_1'] == np.nan or row['lon_1'] == np.nan:
        print('Missing 2')
        return -1.0
    if row['lat_2'] == np.nan or row['lon_2'] == np.nan:
        print('Missing 3')
        return -2.0
    if row['lat_1'] == row['lat_2'] and row['lon_1'] == row['lon_2']:
        # print('Equal')
        return 0.0


    if 0:
        sqrt = math.sqrt((row['lat_1'] - row['lat_2'])*(row['lat_1'] - row['lat_2']) +
                     (row['lon_1'] - row['lon_2'])*(row['lon_1'] - row['lon_2']))
    h = haversine((row['lat_1'], row['lon_1']), (row['lat_2'], row['lon_2']))
    # print(h, sqrt)
    return h


def count_haversine_distance(table, items):
    print("Count parameter...")
    temp1 = items[['itemID', 'lat', 'lon']].rename(columns={
        'itemID': 'itemID_1',
        'lat': 'lat_1',
        'lon': 'lon_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'lat', 'lon']].rename(columns={
        'itemID': 'itemID_2',
        'lat': 'lat_2',
        'lon': 'lon_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table['distance_haversine'] = table.apply(distance_haversine, axis=1)
    table = table.drop(['lat_1'], axis=1)
    table = table.drop(['lat_2'], axis=1)
    table = table.drop(['lon_1'], axis=1)
    table = table.drop(['lon_2'], axis=1)
    print(table['distance_haversine'].describe())
    return table


def get_haversine_distance_train():

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(np.int32),
        'itemID_2': np.dtype(np.int32),
        'isDuplicate': np.dtype(np.int32),
        'generationMethod': np.dtype(np.int32),
    }

    types2 = {
        'itemID': np.dtype(np.int32),
        'categoryID': np.dtype(np.int32),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(np.float64),
        'locationID': np.dtype(np.int32),
        'metroID': np.dtype(np.float32),
        'lat': np.dtype(np.float64),
        'lon': np.dtype(np.float64),
    }

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv"
    pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv",
                        dtype=types2,
                        usecols=['itemID', 'price', 'lat', 'lon'])

    print('Train length: {}'.format(len(pairs.index)))
    train = count_haversine_distance(pairs, items)

    train = train.drop(['isDuplicate'], axis=1)
    train = train.drop(['generationMethod'], axis=1)
    print(train.describe())
    print('Saving train...')
    train.to_csv("../modified_data/haversine_distance_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def get_haversine_distance_test():

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(np.int32),
        'itemID_2': np.dtype(np.int32),
        'id': np.dtype(np.int32),
    }

    types2 = {
        'itemID': np.dtype(np.int32),
        'categoryID': np.dtype(np.int32),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(np.float64),
        'locationID': np.dtype(np.int32),
        'metroID': np.dtype(np.float32),
        'lat': np.dtype(np.float64),
        'lon': np.dtype(np.float64),
    }

    print("Load ItemPairs_test.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../input/ItemPairs_test.csv"
    pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_test.csv",
                        dtype=types2,
                        usecols=['itemID', 'price', 'lat', 'lon'])

    print('Train length: {}'.format(len(pairs.index)))
    train = count_haversine_distance(pairs, items)

    train = train.drop(['id'], axis=1)
    print(train.describe())
    print('Saving test...')
    train.to_csv("../modified_data/haversine_distance_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


# get_haversine_distance_train()
get_haversine_distance_test()
