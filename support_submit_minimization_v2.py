# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize


def get_ensemble_score(x, df_mean):
    x_sum = sum(x)
    if abs(x_sum) < 0.00000001:
        return 1000000000
    df_mean['probability'] = x[0]*df_mean['probability0']
    for i in range(1, len(x)):
        df_mean['probability'] += x[i]*df_mean['probability' + str(i)]
    df_mean['probability'] /= x_sum
    auc = roc_auc_score(df_mean['isDuplicate'].values, df_mean['probability'].values)
    ret_val = 1.0000001 - auc
    print(1.0000001 - ret_val, x)
    return ret_val


def read_tests(fold_tst):
    total = 0
    df_tst = dict()
    for f in fold_tst:
        df_tst[total] = pd.read_csv(f)
        total += 1
    return df_tst


def create_test_subm(fold_tst, koeffs, suffix):
    df_tst = read_tests(fold_tst)
    df_ret = df_tst[0].copy()
    df_ret['probability'] *= koeffs[0]
    for i in range(1, len(df_tst)):
        df_ret['probability'] += df_tst[i]['probability']*koeffs[i]
    df_ret['probability'] /= sum(koeffs)
    print(sum(koeffs))
    print(df_ret.describe())
    if df_ret['probability'].max() >= 1.0:
        print('Error max probability: {}'.format(df_ret['probability'].max()))
        exit()
    if df_ret['probability'].min() <= 0.0:
        print('Error min probability: {}'.format(df_ret['probability'].min()))
        exit()
    df_ret.to_csv("../analysis/merge/for_subm_" + suffix + ".csv", index=False)


def analysis(fold_val):
    total = 0
    df_val = dict()
    for f in fold_val:
        df_val[total] = pd.read_csv(f)
        auc = roc_auc_score(df_val[total]['isDuplicate'].values, df_val[total]['probability'].values)
        print('Auc for experiment {}: {}'.format(total, auc))
        total += 1

    df_mean = df_val[0].copy()
    for i in range(1, len(fold_val)):
        df_mean['probability'] += df_val[i]['probability']
    df_mean['probability'] /= len(fold_val)
    auc = roc_auc_score(df_mean['isDuplicate'].values, df_mean['probability'].values)
    print('Auc for mean: {}'.format(auc))

    alls = []
    x0 = []
    for i in range(0, len(fold_val)):
        val = 'probability' + str(i)
        alls.append(val)
        df_mean[val] = df_val[i]['probability']
        x0.append(1.0)
    df_mean['probability_median'] = df_mean[alls].median(axis=1)
    auc = roc_auc_score(df_mean['isDuplicate'].values, df_mean['probability_median'].values)
    print('Auc for median: {}'.format(auc))

    res = minimize(get_ensemble_score, x0, args=(df_mean), method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
    print(res)
    return res.x

folds = [
    "../analysis/submission_0.97712283726_2016-07-05-23-34_validation.csv",
    "../analysis/submission_0.976041133834_2016-07-05-23-50_validation.csv",
    "../analysis/submission_0.976440354692_2016-07-06-00-18_validation.csv",
]

folds_tst = [
    "../analysis/submission_0.97712283726_2016-07-05-23-34.csv",
    "../analysis/submission_0.976041133834_2016-07-05-23-50.csv",
    "../analysis/submission_0.976440354692_2016-07-06-00-18.csv",
]

koeffs1 = analysis(folds)
create_test_subm(folds_tst, koeffs1, "_minimize")

'''
Auc for experiment 0: 0.9771228372598498
Auc for experiment 1: 0.9760411338342201
Auc for experiment 2: 0.9764403546922159
Auc for mean: 0.9778700972393692
Auc for median: 0.9773485964265286
Auc best: 0.978037639744
[1.4736012,  0.09507276,  1.24990085]
'''

# Mean: 0.94489 Koeffs: 0.94499
