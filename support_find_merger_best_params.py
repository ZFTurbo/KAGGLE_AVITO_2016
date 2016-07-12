# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os
import pickle
import numpy as np

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def get_0_1_val(fl):
    if fl < 0.24:
        return 0
    if fl > 0.76:
        return 1
    return 2


def find_strange_test_pairs(pairs):
    count_1 = 0
    count_0 = 0
    save_index = dict()
    res_1 = dict()
    res_0 = dict()

    # Much faster
    count = 0
    itemID_1 = pairs['itemID_1'].astype(np.int64).copy().values
    itemID_2 = pairs['itemID_2'].astype(np.int64).copy().values
    probability = pairs['probability'].astype(np.float64).copy().values
    for i in range(len(probability)):

        if i % 100000 == 0:
            print('Index: {}'.format(i))

        item_1 = int(itemID_1[i])
        item_2 = int(itemID_2[i])
        isDub = get_0_1_val(probability[i])
        save_index[i] = (item_1, item_2)

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

    print('Total pairs 0: {}'.format(count_0))
    print('Total pairs 1: {}'.format(count_1))
    # Total pairs 0: 1768538
    # Total pairs 1: 1204751
    return res_0, res_1


def find_strange_pairs_small_array(res_0, res_1):
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
    return strange_pairs


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
    perc_incr = 0.07
    perc_decr = 0.4

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

    if prob > 0.51:
        decrease1 = intersect(ind1_elem0, ind2_elem1)
        for i in range(len(decrease1)):
            new_prob = decrease_one_by_percent(new_prob, 0.25)
        decrease1 = intersect(ind2_elem0, ind1_elem1)
        for i in range(len(decrease1)):
            new_prob = decrease_one_by_percent(new_prob, 0.25)
        increase1 = intersect(ind1_elem1, ind2_elem1)
        for i in range(len(increase1)):
            new_prob = increase_one_by_percent(new_prob, 0.07)

        if 0:
            print(id1, id2)
            print(ind1_elem0)
            print(ind1_elem1)
            print(ind2_elem0)
            print(ind2_elem1)
            print(prob, new_prob)

    if prob < 0.49:
        increase1 = intersect(ind1_elem1, ind2_elem1)
        # for i in range(len(increase1)):
        #    new_prob = increase_zero_by_percent(new_prob, 1.0)
        if len(increase1) > 0:
            new_prob = 0.5

    return new_prob


def run_experiment(pairs, num):
    # print(pairs)
    auc_initial = roc_auc_score(pairs['real'].values, pairs['probability'].values)
    print('Auc for experiment: {}'.format(auc_initial))

    res_cache_path = '../data_external/analysis/res_0_1_' + str(num) + '.pickle'
    if not os.path.isfile(res_cache_path):
        res_0, res_1 = find_strange_test_pairs(pairs)
        cache_data((res_0, res_1), res_cache_path)
    else:
        print('Restore res_0_1 from cache...')
        (res_0, res_1) = restore_data(res_cache_path)

    # strange_pairs = find_strange_pairs_small_array(res_0, res_1)
    # print('Strange pairs:', len(strange_pairs))

    if 0:
        pairs['probability_new'] = pairs['probability'].copy()
        count = 0
        for index, row in pairs.iterrows():
            if index % 100000 == 0:
                print('Index: {}'.format(index))

            item_1 = row['itemID_1']
            item_2 = row['itemID_2']
            prob = row['probability']
            new_prob = get_new_probability(item_1, item_2, prob, res_0, res_1)
            if abs(new_prob - prob) < 0.000001:
                count += 1
            pairs.set_value(index, 'probability_new', new_prob)

    # Much faster
    count = 0
    probability_new = pairs['probability'].astype(np.float64).copy().values
    itemID_1 = pairs['itemID_1'].astype(np.int64).copy().values
    itemID_2 = pairs['itemID_2'].astype(np.int64).copy().values
    probability = pairs['probability'].astype(np.float64).copy().values
    for i in range(len(probability_new)):
        if i % 100000 == 0:
            print('Index: {}'.format(i))
        item_1 = int(itemID_1[i])
        item_2 = int(itemID_2[i])
        prob = probability[i]
        new_prob = get_new_probability(item_1, item_2, prob, res_0, res_1)
        if abs(new_prob - prob) > 0.000001:
            count += 1
        probability_new[i] = new_prob
        if i > 100000000:
            exit()


    print('Replacements: ' + str(count))
    auc_new = roc_auc_score(pairs['real'].values, probability_new)
    print('Auc after replace: {}'.format(auc_new))
    improvement = auc_new - auc_initial
    return improvement


def find_mean_score(pairs):
    for i in range(len(pairs)):
        auc_initial = roc_auc_score(pairs[i]['real'].values, pairs[i]['probability'].values)
        print('Independent AUC: {}'.format(auc_initial))

    result = pairs[0].copy()
    result['overall'] = result['probability']
    for i in range(1, len(pairs)):
        result['overall'] += pairs[i]['probability']
    result['overall'] /= len(pairs)
    auc = roc_auc_score(result['real'].values, result['overall'].values)
    print('Mean AUC: {}'.format(auc))
    return auc


def get_new_prob_mean(row, len1):
    val = 0.0
    for i in range(len1):
        val += row['probability_' + str(i)]
    return val


def get_new_prob(row, len1):
    ret_val = 0.0
    all_1 = True
    all_0 = True
    max_val = 0
    min_val = 1
    out = []
    for i in range(len1):
        cur_val = row['probability_' + str(i)]
        out.append(cur_val)
        ret_val += cur_val
        if cur_val > 0.01:
            all_0 = False
        if cur_val < 0.99:
            all_1 = False
        if cur_val > max_val:
            max_val = cur_val
        if cur_val < min_val:
            min_val = cur_val
    ret_val /= len1
    if all_1 == True:
        ret_val = max_val
    if all_0 == True:
        ret_val = min_val

    # out.append(ret_val)
    # if all_1 == True or all_0 == True:
    #    print(out)
    return ret_val


def try_new_method(pairs):
    result = pairs[0].copy()
    for i in range(len(pairs)):
        result['probability_' + str(i)] = pairs[i]['probability']

    print('Start calcs...')
    result['merged_prob'] = result.apply(get_new_prob, args=(len(pairs),), axis=1)
    auc = roc_auc_score(result['real'].values, result['merged_prob'].values)
    print('New AUC: {}'.format(auc))
    return auc


pairs = dict()
for j in range(4):
    pairs_path = '../data_external/analysis/pairs_' + str(j) + '.hdf'
    print('Read from cache!')
    pairs[j] = pd.read_hdf(pairs_path, 'table')

mean_auc = find_mean_score(pairs)
new_auc = try_new_method(pairs)
print('Improvement: {}'.format(new_auc - mean_auc))