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


def get_0_1_val(fl):
    if fl < 0.4:
        return 0
    if fl > 0.6:
        return 1
    return 2


def fix_submission(pairs, index, subm_to_improve):
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
        if index[id] in pairs:
            count += 1
            # print(id, str(prob), str(index[id]))
            if prob > 0.6:
                new_prob = prob * 0.75
            elif prob < 0.4:
                new_prob = prob * 1.25
            else:
                new_prob = prob
                count -= 1
            out.write(id + ',' + str(new_prob) + '\n')
        else:
            out.write(line + '\n')

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
                    if el not in res_1[item_1]:
                        res_1[item_1].append(el)

            # Backward
            if item_2 in res_1:
                if item_1 not in res_1[item_2]:
                    res_1[item_2].append(item_1)
            else:
                res_1[item_2] = [item_1]
            if item_1 in res_1:
                for el in res_1[item_1]:
                    if el not in res_1[item_2]:
                        res_1[item_2].append(el)

        if isDub == 0:
            count_0 += 1
            if item_1 in res_0:
                res_0[item_1].append(item_2)
            else:
                res_0[item_1] = [item_2]
            if item_2 in res_0:
                res_0[item_2].append(item_1)
            else:
                res_0[item_2] = [item_1]

    f.close()
    print('Total pairs 0: {}'.format(count_0))
    print('Total pairs 1: {}'.format(count_1))

    strange_pairs = dict()
    for el in res_1:
        for i in range(len(res_1[el])):
            for j in range(i+1, len(res_1[el])):
                elem1 = res_1[el][i]
                elem2 = res_1[el][j]
                if elem1 in res_0:
                    if elem2 in res_0[elem1]:
                        strange_pairs[(el, elem1)] = 1
                        strange_pairs[(elem1, el)] = 1
                        strange_pairs[(el, elem2)] = 1
                        strange_pairs[(elem2, el)] = 1
                        strange_pairs[(elem1, elem2)] = 1
                        strange_pairs[(elem2, elem1)] = 1
                if elem2 in res_0:
                    if elem1 in res_0[elem2]:
                        strange_pairs[(el, elem1)] = 1
                        strange_pairs[(elem1, el)] = 1
                        strange_pairs[(el, elem2)] = 1
                        strange_pairs[(elem2, el)] = 1
                        strange_pairs[(elem1, elem2)] = 1
                        strange_pairs[(elem2, elem1)] = 1

    print('Strange pairs:', len(strange_pairs))
    fix_submission(strange_pairs, save_index, subm_to_improve)

if __name__ == '__main__':
    find_strange_test_pairs("subm/subm-0.94732.csv")

# LB: 0.93339 LBNEW: 0.93352 Replacements: 6338
# LB: 0.93339 LBNEW: 0.93332 All goes to: 0.5
# LB: 0.93339 LBNEW: 0.93371 All goes to: (0 - 0.2, 1 - 0.8) Replacements: 17431
# LB: 0.93339 LBNEW: 0.93385 All goes to: (0 - 0.3, 1 - 0.7) Fix (0.75, 1.25)  Replacements: 33501
# LB: 0.93339 LBNEW: 0.93376 All goes to: (0 - 0.3, 1 - 0.7) Fix (0.7, 1.3) Replacements: 33501
# LB: 0.93339 LBNEW: 0.93402 All goes to: (0 - 0.4, 1 - 0.6) Fix (0.75, 1.25) Replacements: 49429
# LB: 0.93999 LBNEW: 0.94050 All goes to: (0 - 0.4, 1 - 0.6) Fix (0.75, 1.25) Replacements: 44148
# LB: 0.94016 LBNEW: 0.94065 All goes to: (0 - 0.45, 1 - 0.55) Fix (0.75, 1.25) Replacements: 52273
