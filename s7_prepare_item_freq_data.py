# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import defaultdict
import time
import copy
import json
import glob
import hashlib
import os
import pickle
import imagehash
from PIL import Image
import json
import math

def get_item_freq_table(type = 'train'):
    ids_array = dict()
    if type == 'train':
        # f = open("../input/ItemPairs_train.csv")
        f = open("../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")
    else:
        f = open("../input/ItemPairs_test.csv")

    # Get ID stats
    first_line = f.readline()
    total = 0
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        if type == 'train':
            id1 = int(arr[0])
            id2 = int(arr[1])
        else:
            id1 = int(arr[1])
            id2 = int(arr[2])
        if id1 in ids_array:
            ids_array[id1] += 1
        else:
            ids_array[id1] = 1
        if id2 in ids_array:
            ids_array[id2] += 1
        else:
            ids_array[id2] = 1
        total += 1

    f.close()

    # Get max & avg
    max1 = 0
    avg = 0
    for el in ids_array:
        if ids_array[el] > max1:
            max1 = ids_array[el]
        avg += ids_array[el]
    avg /= len(ids_array)

    print('Max comaprsions for id [{}]: {}'.format(type, max1))
    print('Avg comaprsions for id [{}]: {}'.format(type, avg))

    # Print feature
    if type == 'train':
        # f = open("../input/ItemPairs_train.csv")
        f = open("../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")
    else:
        f = open("../input/ItemPairs_test.csv")

    out = open("../modified_data/" + type + "_IDs_features.csv", "w")
    out.write("itemID_1,itemID_2,ids_freq_id1,ids_freq_id2,ids_freq_diff1,ids_freq_diff2,ids_substruction_abs,ids_equality\n")
    first_line = f.readline()
    total = 0
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        if type == 'train':
            id1 = int(arr[0])
            id2 = int(arr[1])
        else:
            id1 = int(arr[1])
            id2 = int(arr[2])
        ids_equality = 0
        if ids_array[id1] > ids_array[id2]:
            ids_equality = 1
        if ids_array[id1] < ids_array[id2]:
            ids_equality = -1
        diff1 = ids_array[id1]/ids_array[id2]
        diff2 = ids_array[id2]/ids_array[id1]
        ids_substruction_abs = abs(ids_array[id1] - ids_array[id2])
        out.write(str(id1) + ',' + str(id2) + ',' + str(ids_array[id1]) + ',' + str(ids_array[id2]))
        out.write(',' + str(diff1) + ',' + str(diff2) + ',' + str(ids_substruction_abs) + ',' + str(ids_equality))
        out.write('\n')

    return ids_array


def check_correlation():
    # table1 = pd.read_csv("../input/ItemPairs_train.csv")
    table1 = pd.read_csv("../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")
    add_table = pd.read_csv("../modified_data/train_IDs_features.csv")
    table = pd.merge(table1, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    # for f in list(add_table.columns.values):
    #    if f == 'itemID_1' or f == 'itemID_2':
    #        continue
    #    corr = table[f].corrwith(table['isDuplicate'])
    corr = table.corrwith(table['isDuplicate'])
    print(corr)


if 1:
    d1 = get_item_freq_table(type='train')
    d2 = get_item_freq_table(type='test')
    print('Train unique ID: ', str(len(d1)))
    print('Test unique ID: ', str(len(d2)))

    if 0:
        for el in d1:
            if el in d2:
                print("Exists: ", el)

check_correlation()