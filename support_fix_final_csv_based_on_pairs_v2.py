# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import zipfile
import numpy as np
from collections import defaultdict
import pickle
np.random.seed(2016)


def read_submission(subm_to_improve):
    print('Read submission...')
    ret = dict()
    f = open(subm_to_improve)
    first_line = f.readline()
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        id = arr[0]
        prob = float(arr[1])
        ret[id] = prob
    f.close()
    return ret


def intersect(a, b):
    return list(set(a) & set(b))


def decrease_one_by_percent(prob, perc):
    return prob - (prob - 0.5)*perc


def increase_one_by_percent(prob, perc):
    return prob + (1 - prob)*perc


def increase_zero_by_percent(prob, perc):
    return prob + (0.5 - prob)*perc


def get_new_probability(id1, id2, prob, res_0, res_1):
    new_prob = prob

    ind1_elem0 = []
    if id1 in res_0:
       ind1_elem0 = res_0[id1]
    ind1_elem1 = []
    if id1 in res_1:
       ind1_elem1 = res_1[id1]
    ind2_elem0 = []
    if id2 in res_0:
       ind2_elem0 = res_0[id2]
    ind2_elem1 = []
    if id2 in res_1:
       ind2_elem1 = res_1[id2]

    if prob > 0.6:
        decrease1 = intersect(ind1_elem0, ind2_elem1)
        for i in range(len(decrease1)):
            new_prob = decrease_one_by_percent(new_prob, 0.1)
        decrease1 = intersect(ind2_elem0, ind1_elem1)
        for i in range(len(decrease1)):
            new_prob = decrease_one_by_percent(new_prob, 0.1)
        increase1 = intersect(ind1_elem1, ind2_elem1)
        for i in range(len(increase1)):
            new_prob = increase_one_by_percent(new_prob, 0.05)

        if 0:
            print(id1, id2)
            print(ind1_elem0)
            print(ind1_elem1)
            print(ind2_elem0)
            print(ind2_elem1)
            print(prob, new_prob)

    if prob < 0.4:
        increase1 = intersect(ind1_elem1, ind2_elem1)
        for i in range(len(increase1)):
            new_prob = increase_zero_by_percent(new_prob, 0.1)

    return new_prob


def get_0_1_val(fl):
    if fl < 0.3:
        return 0
    if fl > 0.7:
        return 1
    return 2


def fix_submission(res_0, res_1, index, subm_to_improve):
    print('Rewrite submission...')
    out_file = subm_to_improve + "-fixed.csv"
    out = open(out_file, "w")
    f = open(subm_to_improve)
    first_line = f.readline()
    out.write(first_line)
    count = 0
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        id = arr[0]
        prob = float(arr[1])
        item_1 = int(index[id][0])
        item_2 = int(index[id][1])
        new_prob = get_new_probability(item_1, item_2, prob, res_0, res_1)
        if abs(new_prob - prob) > 0.000001:
            count += 1
        out.write(id + ',' + str(new_prob) + '\n')

    print('Replacements: ' + str(count))
    f.close()
    out.close()

    print('Creating zip-file...')
    z = zipfile.ZipFile(out_file + ".zip", "w", zipfile.ZIP_DEFLATED)
    z.write(out_file)
    z.close()


def find_strange_test_pairs(subm_to_improve):
    count_1 = 0
    count_0 = 0
    subm = read_submission(subm_to_improve)
    save_index = dict()
    res_1 = dict()
    res_0 = dict()
    f = open("../input/ItemPairs_test.csv")
    first_line = f.readline()
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        id = arr[0]
        item_1 = int(arr[1])
        item_2 = int(arr[2])
        isDub = get_0_1_val(subm[id])
        save_index[id] = (item_1, item_2)

        if isDub == 1:
            # Forward
            count_1 += 1
            if item_1 in res_1:
                if item_2 not in res_1[item_1]:
                    res_1[item_1].append(item_2)
            else:
                res_1[item_1] = [item_2]
            if item_2 in res_1:
                for el in res_1[item_2]:
                    if el not in res_1[item_1] and el != item_1:
                        res_1[item_1].append(el)

            # Backward
            if item_2 in res_1:
                if item_1 not in res_1[item_2]:
                    res_1[item_2].append(item_1)
            else:
                res_1[item_2] = [item_1]
            if item_1 in res_1:
                for el in res_1[item_1]:
                    if el not in res_1[item_2] and el != item_2:
                        res_1[item_2].append(el)

        if isDub == 0:
            count_0 += 1
            if item_1 in res_0:
                if item_2 not in res_0[item_1]:
                    res_0[item_1].append(item_2)
            else:
                res_0[item_1] = [item_2]
            if item_2 in res_0:
                if item_1 not in res_0[item_2]:
                    res_0[item_2].append(item_1)
            else:
                res_0[item_2] = [item_1]

    f.close()
    print('Total pairs 0: {}'.format(count_0))
    print('Total pairs 1: {}'.format(count_1))

    fix_submission(res_0, res_1, save_index, subm_to_improve)

if __name__ == '__main__':
    find_strange_test_pairs("subm/subm-0.94700.csv")

# OLD: 0.94732 NEW: 0.94635
# OLD: 0.94700 NEW: 0.94655