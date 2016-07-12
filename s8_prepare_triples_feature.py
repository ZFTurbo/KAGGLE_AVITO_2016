# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import zipfile
import numpy as np
from collections import defaultdict
import pickle
np.random.seed(2016)


def intersect(a, b):
    return list(set(a) & set(b))


def create_pairs_array(type='train'):
    pairs = dict()
    if type == 'train':
        f = open("../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")
    else:
        f = open("../input/ItemPairs_test.csv")
    first_line = f.readline()
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        if type == 'train':
            item_1 = int(arr[0])
            item_2 = int(arr[1])
        else:
            item_1 = int(arr[1])
            item_2 = int(arr[2])

        if item_1 not in pairs:
            pairs[item_1] = [item_2]
        else:
            pairs[item_1].append(item_2)
        if item_2 not in pairs:
            pairs[item_2] = [item_1]
        else:
            pairs[item_2].append(item_1)
    f.close()

    # Write feature
    if type == 'train':
        out = open("../modified_data/triples_exists_train.csv", "w")
        f = open("../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")
    else:
        out = open("../modified_data/triples_exists_test.csv", "w")
        f = open("../input/ItemPairs_test.csv")
    first_line = f.readline()
    out.write("itemID_1,itemID_2,triple_exists\n")
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        if type == 'train':
            item_1 = int(arr[0])
            item_2 = int(arr[1])
        else:
            item_1 = int(arr[1])
            item_2 = int(arr[2])
        lst = intersect(pairs[item_1], pairs[item_2])
        # print(len(lst), lst)
        prnt = str(len(lst))
        if type == 'train':
            out.write(arr[0] + ',' + arr[1] + ',' + prnt + '\n')
        else:
            out.write(arr[1] + ',' + arr[2] + ',' + prnt + '\n')

    f.close()
    out.close()


if __name__ == '__main__':
    create_pairs_array('train')
    create_pairs_array('test')

