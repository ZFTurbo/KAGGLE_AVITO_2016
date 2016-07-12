# -*- coding: utf-8 -*-

import os
import zipfile
import numpy as np
from collections import defaultdict
import pickle
np.random.seed(2016)

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def get_new_pairs(res, el1, el2, new_pairs):
    pairs = []
    for elem in res[el2]:
        if elem != el1:
            if el1 in res:
                if elem in res[el1]:
                    continue
            if (el1, elem) not in new_pairs:
                if (el1, elem) not in pairs:
                    pairs.append((el1, elem))
    return pairs


def find_additional_pairs():
    res_cache = "cache/additional_pairs_search1.pickle.dat"
    count_1 = 0
    count_0 = 0
    if not os.path.isfile(res_cache) or 1:
        res_1 = dict()
        res_0 = dict()
        # f = open("../modified_data/ItemPairs_train_removed_errors.csv")
        f = open("../input/ItemPairs_train.csv")
        first_line = f.readline()
        while 1:
            line = f.readline().strip()
            if line == '':
                break
            arr = line.split(',')
            item_1 = int(arr[0])
            item_2 = int(arr[1])
            isDub = int(arr[2])

            if isDub == 1:
                count_1 += 1
                if item_1 in res_1:
                    res_1[item_1].append(item_2)
                else:
                    res_1[item_1] = [item_2]
            if isDub == 0:
                count_0 += 1
                if item_1 in res_0:
                    res_0[item_1].append(item_2)
                else:
                    res_0[item_1] = [item_2]

        cache_data((res_0, res_1), res_cache)
        f.close()
        print('Total pairs 0: {}'.format(count_0))
        print('Total pairs 1: {}'.format(count_1))
    else:
        (res_0, res_1) = restore_data(res_cache)

    # Pairs type (ID_1, ID_2 = EQUAL, ID_2, ID_3 = EQUAL => ID_1, ID_3 => EQUAL)
    # Add if not exists in train dataset
    new_pairs_1 = []
    count_multiple_1 = 0
    for el in res_1:
        for i in range(len(res_1[el])):
            if res_1[el][i] in res_1:
                tpl = get_new_pairs(res_1, el, res_1[el][i], new_pairs_1)
                new_pairs_1 += tpl
                # print(el, res[el])
                # print(res[el][i], res[res[el][i]])
                # exit()
                count_multiple_1 += 1

    print('Total elements 1: {}'.format(len(res_1)))
    print('Pairs more than 1: {}'.format(count_multiple_1))
    print('Length of New Pair List 1: {}'.format(len(new_pairs_1)))

    # Pairs type (ID_1, ID_2 = EQUAL, ID_2, ID_3 = NOT EQUAL => ID_1, ID_3 => NOT EQUAL)
    # Add if not exists in train dataset
    new_pairs_0 = []
    count_multiple_0 = 0
    for el in res_1:
        for i in range(len(res_1[el])):
            if res_1[el][i] in res_0:
                tpl = get_new_pairs(res_0, el, res_1[el][i], new_pairs_0)
                new_pairs_0 += tpl
                # print(el, res[el])
                # print(res[el][i], res[res[el][i]])
                # exit()
                count_multiple_0 += 1

    print('Total elements 0: {}'.format(len(res_0)))
    print('Pairs more than 0: {}'.format(count_multiple_0))
    print('Length of New Pair List 0: {}'.format(len(new_pairs_0)))

    new_pairs_0 = sorted(new_pairs_0)
    print(new_pairs_0[0:100])
    new_pairs_1 = sorted(new_pairs_1)
    print(new_pairs_1[0:100])
    return new_pairs_0, new_pairs_1


def create_new_train_pairs_file(pairs_0, pairs_1):
    count_write_0 = 0
    count_write_1 = 0
    res_0 = dict(pairs_0)
    res_1 = dict(pairs_1)
    print('Unique dict 0: {}'.format(len(res_0)))
    print('Unique dict 1: {}'.format(len(res_1)))
    # f = open("../modified_data/ItemPairs_train_removed_errors.csv")
    f = open("../input/ItemPairs_train.csv")
    out = open("../modified_data/ItemPairs_train_with_additional_pairs.csv", "w")
    first_line = f.readline()
    out.write(first_line)
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        out.write(line + '\n')
        arr = line.split(',')
        item_1 = int(arr[0])
        item_2 = int(arr[1])
        isDub = int(arr[2])

        if item_1 in res_0:
            save = []
            for p in pairs_0:
                if p[0] == item_1:
                    count_write_0 += 1
                    out.write('{},{},{},4\n'.format(p[0], p[1], 0))
                    save.append(p)
            for s in save:
                pairs_0.remove(s)
        if item_1 in res_1:
            save = []
            for p in pairs_1:
                if p[0] == item_1:
                    count_write_1 += 1
                    out.write('{},{},{},4\n'.format(p[0], p[1], 1))
                    save.append(p)
            for s in save:
                pairs_1.remove(s)

    print('Printed new pairs 0: {}'.format(count_write_0))
    print('Printed new pairs 1: {}'.format(count_write_1))

    out.close()
    f.close()


def check_pairs(pairs):
    for el in pairs:
        e1 = el[0]
        e2 = el[1]
        if int(e2) < int(e1):
            print('Check some problem here! {} and {}'.format(e1, e2))


def check_dublicates(path):
    f = open(path, "r")
    data = dict()
    f.readline()
    count = 0
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        item_1 = int(arr[0])
        item_2 = int(arr[1])
        el = (item_1, item_2)
        if el in data:
            print(el)
            count += 1
        else:
            data[el] = 1

    print('Dublicates: {}'.format(count))
    # 7300
    f.close()


def find_strange_pairs():
    res_cache = "cache/additional_pairs_search.pickle.dat"
    count_1 = 0
    count_0 = 0
    if not os.path.isfile(res_cache) or 1:
        res_1 = dict()
        res_0 = dict()
        f = open("../input/ItemPairs_train.csv")
        first_line = f.readline()
        while 1:
            line = f.readline().strip()
            if line == '':
                break
            arr = line.split(',')
            item_1 = int(arr[0])
            item_2 = int(arr[1])
            isDub = int(arr[2])

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

        cache_data((res_0, res_1), res_cache)
        f.close()
        print('Total pairs 0: {}'.format(count_0))
        print('Total pairs 1: {}'.format(count_1))
    else:
        (res_0, res_1) = restore_data(res_cache)

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

    f = open("../input/ItemPairs_train.csv")
    out = open("../modified_data/ItemPairs_train_removed_errors.csv", "w")
    first_line = f.readline()
    out.write(first_line)
    count = 0
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        item_1 = int(arr[0])
        item_2 = int(arr[1])
        isDub = int(arr[2])

        if (item_1, item_2) not in strange_pairs and (item_2, item_1) not in strange_pairs:
            out.write(line + '\n')
        else:
            count += 1

    print('Removed lines: {}'.format(count))
    out.close()
    f.close()

    # v1
    # Strange pairs: 18246
    # Removed lines: 9123
    # v2
    # Strange pairs: 47432
    # Removed lines: 23716
    # v3
    # Strange pairs: 69394
    # Removed lines: 29430


def fix_csv(path, out_path):
    check_array = dict()
    f = open(path)
    out = open(out_path, "w")
    first_line = f.readline()
    out.write(first_line)
    count = 0
    while 1:
        line = f.readline()
        if line == '':
            break
        arr = line.split(',')
        item_1 = int(arr[0])
        item_2 = int(arr[1])

        if (item_1, item_2) not in check_array:
            check_array[(item_1, item_2)] = 1
        else:
            check_array[(item_1, item_2)] += 1
            count += 1
    f.close()

    print('Lines to remove: {}'.format(count))
    f = open(path)
    first_line = f.readline()
    count = 0
    while 1:
        line = f.readline()
        if line == '':
            break
        arr = line.split(',')
        item_1 = int(arr[0])
        item_2 = int(arr[1])
        gen = int(arr[3])

        if gen != 4 or check_array[(item_1, item_2)] == 1:
            out.write(line)
        else:
            count += 1

    print('Lines removed: {}'.format(count))
    out.close()
    f.close()


if __name__ == '__main__':
    if 0:
        strange_pairs = find_strange_pairs()
        check_dublicates("../modified_data/ItemPairs_train_removed_errors.csv")
    if 0:
        pairs_0, pairs_1 = find_additional_pairs()
        check_pairs(pairs_0)
        check_pairs(pairs_1)
        create_new_train_pairs_file(pairs_0, pairs_1)
        check_dublicates("../modified_data/ItemPairs_train_with_additional_pairs.csv")
    fix_csv("../modified_data/ItemPairs_train_with_additional_pairs.csv", "../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")
    check_dublicates("../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv")

