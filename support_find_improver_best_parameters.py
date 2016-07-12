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

        # if i % 100000 == 0:
        #    print('Index: {}'.format(i))

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

    if prob > 0.55:
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

    if prob < 0.45:
        increase1 = intersect(ind1_elem1, ind2_elem1)
        for i in range(len(increase1)):
           new_prob = increase_zero_by_percent(new_prob, 0.1)
        # if len(increase1) > 0:
        #    new_prob = 0.5

    return new_prob


def run_experiment(pairs, num):
    # print(pairs)
    auc_initial = roc_auc_score(pairs['real'].values, pairs['probability'].values)
    print('Auc for experiment: {}'.format(auc_initial))

    res_cache_path = '../data_external/analysis/res_0_1_' + str(num) + '.pickle'
    if not os.path.isfile(res_cache_path) or 1:
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
        # if i % 100000 == 0:
            # print('Index: {}'.format(i))
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


fold_1_val = [
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv",
]
fold_1_labels = "../data_external/analysis/FOLD1/labels_val.csv"

fold_2_val = [
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv",
]
fold_2_labels = "../data_external/analysis/FOLD2/labels_val.csv"

fold_3_val = [
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv",
]
fold_3_labels = "../data_external/analysis/FOLD3/labels_val.csv"

if 0:
    for j in range(4):
        pairs_path = '../data_external/analysis/pairs_' + str(j) + '.hdf'
        if not os.path.isfile(pairs_path):
            pairs = pd.concat([
                pd.read_csv("../input/ItemPairs_train.csv"),
                pd.read_csv("../modified_data/ItemPairs_zft.csv")
            ])
            pairs['elems'] = range(len(pairs.index))

            df_val = dict()
            total = 0
            df_real = pd.read_csv(fold_3_labels)
            df_val[total] = pd.read_csv(fold_3_val[j])
            df_val[total] = pd.merge(df_val[total], df_real, how='left', on=['id'], left_index=True)
            total += 1
            df_real = pd.read_csv(fold_2_labels)
            df_val[total] = pd.read_csv(fold_2_val[j])
            df_val[total] = pd.merge(df_val[total], df_real, how='left', on=['id'], left_index=True)
            total += 1
            df_real = pd.read_csv(fold_1_labels)
            df_val[total] = pd.read_csv(fold_1_val[j])
            df_val[total] = pd.merge(df_val[total], df_real, how='left', on=['id'], left_index=True)
            total += 1

            iterator = restore_data("../data_external/analysis/skf_3_777_890d1a734abac3f79cd31273afbcfcb53104fdc3.pickle")
            total = 0
            for itrn, ival in iterator:
                df_val[total]['elems'] = pd.Series(ival, index=df_val[total].index)
                total += 1

            df_val_overall = pd.concat([
                df_val[0],
                df_val[1],
                df_val[2],
            ])
            # print(df_val_overall)
            pairs = pd.merge(pairs, df_val_overall[['probability', 'elems', 'real']], how='left', on=['elems'], left_index=True)
            pairs = pairs.reset_index(drop=True)
            pairs.to_hdf(pairs_path, 'table', format='t', complevel=9, complib='blosc')
        else:
            print('Read from cache!')
            pairs = pd.read_hdf(pairs_path, 'table')
        improvement = run_experiment(pairs, j)
        print('Improvement: {}'.format(improvement))

if 0:
    pairs = pd.read_csv("../run_0.94453/submission_0.997707203673_2016-07-09-02-41_train.csv")
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 5)
    print('Improvement: {}'.format(improvement))

    pairs = pd.read_csv("../run_0.94292/submission_0.996118183392_2016-07-09-03-16_train.csv")
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 6)
    print('Improvement: {}'.format(improvement))

    pairs = pd.read_csv("../run_0.94272/submission_0.993003724746_2016-07-09-10-54_train.csv")
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 7)
    print('Improvement: {}'.format(improvement))

if 1:
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/04_col-0.8_eta-0.01_max-13_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/pairs_val.csv')
    f1 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/04_col-0.8_eta-0.01_max-13_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/pairs_val.csv')
    f2 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/04_col-0.8_eta-0.01_max-13_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/pairs_val.csv')
    f3 = pd.concat([f, pr], axis=1)
    pairs = pd.concat([f1, f2, f3], axis=0)
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 8)
    print('Improvement: {}'.format(improvement))


    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/04_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/pairs_val.csv')
    f1 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/04_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/pairs_val.csv')
    f2 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/04_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/pairs_val.csv')
    f3 = pd.concat([f, pr], axis=1)
    pairs = pd.concat([f1, f2, f3], axis=0)
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 8)
    print('Improvement: {}'.format(improvement))


    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/04_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/pairs_val.csv')
    f1 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/04_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/pairs_val.csv')
    f2 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/04_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/pairs_val.csv')
    f3 = pd.concat([f, pr], axis=1)
    pairs = pd.concat([f1, f2, f3], axis=0)
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 8)
    print('Improvement: {}'.format(improvement))


    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/04_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/pairs_val.csv')
    f1 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/04_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/pairs_val.csv')
    f2 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/04_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/pairs_val.csv')
    f3 = pd.concat([f, pr], axis=1)
    pairs = pd.concat([f1, f2, f3], axis=0)
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 8)
    print('Improvement: {}'.format(improvement))


    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/04_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD1/pairs_val.csv')
    f1 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/04_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD2/pairs_val.csv')
    f2 = pd.concat([f, pr], axis=1)
    f = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/04_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv')
    pr = pd.read_csv('../data_external/analysis2/ensemble-v8-validations_04/FOLD3/pairs_val.csv')
    f3 = pd.concat([f, pr], axis=1)
    pairs = pd.concat([f1, f2, f3], axis=0)
    pairs['real'] = pairs['isDuplicate']
    improvement = run_experiment(pairs, 8)
    print('Improvement: {}'.format(improvement))



f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/04_col-0.8_eta-0.01_max-13_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/pairs_val.csv')
f1 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/04_col-0.8_eta-0.01_max-13_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/pairs_val.csv')
f2 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/04_col-0.8_eta-0.01_max-13_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/pairs_val.csv')
f3 = pd.concat([f, pr], axis=1)
pairs = pd.concat([f1, f2, f3], axis=0)
pairs['real'] = pairs['isDuplicate']
improvement = run_experiment(pairs, 8)
print('Improvement: {}'.format(improvement))


f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/04_col-0.8_eta-0.02_max-11_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/pairs_val.csv')
f1 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/04_col-0.8_eta-0.02_max-11_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/pairs_val.csv')
f2 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/04_col-0.8_eta-0.02_max-11_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/pairs_val.csv')
f3 = pd.concat([f, pr], axis=1)
pairs = pd.concat([f1, f2, f3], axis=0)
pairs['real'] = pairs['isDuplicate']
improvement = run_experiment(pairs, 8)
print('Improvement: {}'.format(improvement))


f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/04_col-0.8_eta-0.03_max-10_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/pairs_val.csv')
f1 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/04_col-0.8_eta-0.03_max-10_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/pairs_val.csv')
f2 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/04_col-0.8_eta-0.03_max-10_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/pairs_val.csv')
f3 = pd.concat([f, pr], axis=1)
pairs = pd.concat([f1, f2, f3], axis=0)
pairs['real'] = pairs['isDuplicate']
improvement = run_experiment(pairs, 8)
print('Improvement: {}'.format(improvement))


f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/04_col-0.8_eta-0.04_max-9_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/pairs_val.csv')
f1 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/04_col-0.8_eta-0.04_max-9_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/pairs_val.csv')
f2 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/04_col-0.8_eta-0.04_max-9_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/pairs_val.csv')
f3 = pd.concat([f, pr], axis=1)
pairs = pd.concat([f1, f2, f3], axis=0)
pairs['real'] = pairs['isDuplicate']
improvement = run_experiment(pairs, 8)
print('Improvement: {}'.format(improvement))


f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/04_col-0.8_eta-0.015_max-12_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD1/pairs_val.csv')
f1 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/04_col-0.8_eta-0.015_max-12_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD2/pairs_val.csv')
f2 = pd.concat([f, pr], axis=1)
f = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/04_col-0.8_eta-0.015_max-12_min-6_sub-0.9_val.csv')
pr = pd.read_csv('../data_external/analysis2/ensemble-v9-validations_04/FOLD3/pairs_val.csv')
f3 = pd.concat([f, pr], axis=1)
pairs = pd.concat([f1, f2, f3], axis=0)
pairs['real'] = pairs['isDuplicate']
improvement = run_experiment(pairs, 8)
print('Improvement: {}'.format(improvement))

# First version:
# 0 - 0.35, 1 - 0.65, FIX: 0.95 and 1.05: OLD: 0.967505 NEW: 0.967579 Improvement = 0.000074
# 0 - 0.45, 1 - 0.55, FIX: 0.9 and 1.1: OLD: 0.967505 NEW: 0.967600 Improvement = 0.000095
# 0 - 0.45, 1 - 0.55, FIX: set all as 0.5: OLD: 0.967505 NEW: 0.9668866 Improvement = -0.0006184

# Next version:
# 0.1 Perc, 0.55, 0.45 = Improvement: 0.00021097
# 0.1 Perc, 0.51, 0.49 = Improvement: 0.00021236
# 0.2 Perc, 0.51, 0.49 = Improvement: -0.000458381
# 0.15 Perc, 0.51, 0.49 = Improvement: -0.00010411
# 0.11 Perc, 0.51, 0.49 = Improvement: 0.00015822
# 0.09 Perc, 0.51, 0.49 = Improvement: 0.00025969
# 0.02 Perc, 0.51, 0.49 = Improvement: 0.00027759
# 0.04 Perc, 0.51, 0.49 = Improvement: 0.00035211
# 0.06 Perc, 0.51, 0.49 = Improvement: 0.00035040
# 0.05 Perc, 0.51, 0.49 (0 - 0.49, 1 - 0.51) = Improvement: 0.00031061
# 0.05 Perc, 0.51, 0.49 (0 - 0.45, 1 - 0.55) = Improvement: 0.00035858
# 0.05 Perc, 0.51, 0.49 (0 - 0.4, 1 - 0.6) = Improvement: 0.00039856
# 0.05 Perc, 0.51, 0.49 (0 - 0.35, 1 - 0.65) = Improvement: 0.00041637
# 0.05 Perc, 0.51, 0.49 (0 - 0.3, 1 - 0.7) = Improvement: 0.00042379 (Replacements: 402800)
# 0.05 Perc, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00043069 (Replacements: 380255)
# 0.05 Perc, 0.51, 0.49 (0 - 0.2, 1 - 0.8) = Improvement: 0.00040609 (Replacements: 355365) - ухудшение
# 0.05 Perc, 0.55, 0.45 (0 - 0.25, 1 - 0.75) = Improvement: 0.00041327 (Replacements: 375379) - ухудшение
# 0.1 Perc, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00044151 (Replacements: 380254)
# 0.1 Perc, 0.51, 0.49 (0 - 0.25, 1 - 0.75) Removed increase 1 part = Improvement: 0.00026610 (Replacements: 51038) - ухудшение
# 0.1 Perc Decr 0.15 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00028382 (Replacements: 380255) - ухудшение
# 0.1 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00049999 (Replacements: 380255)
# 0.15 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00053209
# 0.15 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00053209
# 0.2 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00054971
# 0.3 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00056552
# 0.5 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00056474
# 0.35 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00056814
# 0.45 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00056726
# 0.4 Perc Decr 0.05 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00056853
# 0.4 Perc Decr 0.03 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00050001
# 0.4 Perc Decr 0.06 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00058272
# 0.4 Perc Decr 0.07 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00058683
# 0.4 Perc Decr 0.08 Incr, 0.51, 0.49 (0 - 0.25, 1 - 0.75) = Improvement: 0.00058255
# 0.4 Perc Decr 0.07 Incr, 0.51, 0.49 (0 - 0.3, 1 - 0.7) = Improvement: 0.00054415
# 0.4 Perc Decr 0.07 Incr, 0.51, 0.49 (0 - 0.26, 1 - 0.74) = Improvement: 0.00058329
# 0.4 Perc Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00058990
# 0.4 Perc Decr 0.07 Incr, 0.51, 0.49 (0 - 0.23, 1 - 0.77) = Improvement: 0.00058334
# 0.4 Perc 0, 0.3 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00059756
# 0.4 Perc 0, 0.2 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00059898
# 0.4 Perc 0, 0.1 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00058484
# 0.4 Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00059935
# 0.5 Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00061220
# 0.6 Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00062214
# 0.7 Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00062991
# 0.8 Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00063589
# 0.9 Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00064046
# 1.0 (Const = 0.5) Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00064427
# Const = 0.6 (P0) 0.25 (P1 Decr) 0.07 (P1 Incr), 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00063972
# Const = 0.49 (P0) 0.25 (P1 Decr) 0.07 (P1 Incr), 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00064248
# Const = 0.51 (P0) 0.25 (P1 Decr) 0.07 (P1 Incr), 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00064638
# 1.0 (Const = 0.5) Perc 0, 0.25 Perc 1, Decr 0.07 Incr, 0.51, 0.49 (0 - 0.24, 1 - 0.76) = Improvement: 0.00064427 (0.00064418, 0.00065728, 0.00067111)