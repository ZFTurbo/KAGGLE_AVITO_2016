# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize


def get_ensemble_score(x, df_mean):
    x_sum = sum(x)
    if abs(x_sum) < 0.00000001:
        return 1000000000
    df_mean['probability'] = (x[0]*df_mean['probability0']
                              + x[1]*df_mean['probability1']
                              + x[2]*df_mean['probability2']
                              + x[3]*df_mean['probability3'])/x_sum
    auc = roc_auc_score(df_mean['real'].values, df_mean['probability'].values)
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
    df_ret.to_csv("../data_external/analysis/for_subm_" + suffix + ".csv", index=False)


def analysis(fold_val, fold_labels):
    if fold_labels[-1:] == 'e':
        df1_real = pd.read_pickle(fold_labels)
    else:
        df1_real = pd.read_csv(fold_labels)
    total = 0
    df_val = dict()
    for f in fold_val:
        df_val[total] = pd.read_csv(f)
        df_val[total] = pd.merge(df_val[total], df1_real, how='left', on=['id'], left_index=True)
        # sub = (df_val[total]['real'] - df_val[total]['probability']).abs()
        # avg_error = sub.mean()
        auc = roc_auc_score(df_val[total]['real'].values, df_val[total]['probability'].values)
        print('Auc for experiment {}: {}'.format(total, auc))
        total += 1

    df_mean = df_val[0].copy()
    df_mean['probability'] = (df_val[0]['probability'] + df_val[1]['probability'] + df_val[2]['probability'] + df_val[3]['probability'])/4
    auc = roc_auc_score(df_mean['real'].values, df_mean['probability'].values)
    print('Auc for mean: {}'.format(auc))

    df_mean['probability0'] = df_val[0]['probability']
    df_mean['probability1'] = df_val[1]['probability']
    df_mean['probability2'] = df_val[2]['probability']
    df_mean['probability3'] = df_val[3]['probability']
    df_mean['probability_median'] = df_mean[['probability0', 'probability1', 'probability2', 'probability3']].median(axis=1)
    # print(df_mean['probability_median'].describe())
    # print(len(df_mean['probability_median'].index))
    auc = roc_auc_score(df_mean['real'].values, df_mean['probability_median'].values)
    print('Auc for median: {}'.format(auc))

    x0 = [1.0, 1.0, 1.0, 1.0]
    res = minimize(get_ensemble_score, x0, args=(df_mean), method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
    print(res)
    return res.x

fold_1_val = [
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv",
]
fold_1_tst = [
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD1/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_tst.csv",
]
fold_1_labels = "../data_external/analysis/FOLD1/labels_val.csv"

fold_2_val = [
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv",
]
fold_2_tst = [
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD2/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_tst.csv",
]
fold_2_labels = "../data_external/analysis/FOLD2/labels_val.csv"

fold_3_val = [
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_val.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_val.csv",
]
fold_3_tst = [
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.02_max-12_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.03_max-11_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.04_max-10_min-6_sub-0.9_tst.csv",
    "../data_external/analysis/FOLD3/03_col-0.8_eta-0.05_max-9_min-6_sub-0.9_tst.csv",
]
fold_3_labels = "../data_external/analysis/FOLD3/labels_val.csv"

# koeffs1 = analysis(fold_1_val, fold_1_labels)
# koeffs2 = analysis(fold_2_val, fold_2_labels)
# koeffs3 = analysis(fold_3_val, fold_3_labels)

if 0:
    # k1
    create_test_subm(fold_1_tst, [1.97425927,  1.30212844,  0.57839624, -0.05058898], "_1_minimize")
    create_test_subm(fold_1_tst, [1.0, 1.0, 1.0, 1.0], "_1_mean")
    # k2
    create_test_subm(fold_2_tst, [1.93942601,  1.43086271,  0.58615896, -0.10463731], "_2_minimize")
    create_test_subm(fold_2_tst, [1.0, 1.0, 1.0, 1.0], "_2_mean")
    # k3
    # 0.967832317553 vs 0.9677213878703272
    create_test_subm(fold_3_tst, [2.11495449,  1.98401621,  0.75626197, -0.09433551], "_3_minimize")
    create_test_subm(fold_3_tst, [1.0, 1.0, 1.0, 1.0], "_3_mean")

fold_mean_tst = [
    "../data_external/analysis/for_subm__1_mean.csv",
    "../data_external/analysis/for_subm__2_mean.csv",
    "../data_external/analysis/for_subm__3_mean.csv",
]
fold_minimize_tst = [
    "../data_external/analysis/for_subm__1_minimize.csv",
    "../data_external/analysis/for_subm__2_minimize.csv",
    "../data_external/analysis/for_subm__3_minimize.csv",
]

create_test_subm(fold_mean_tst, [1.0, 1.0, 1.0], "_mean_overall")
create_test_subm(fold_minimize_tst, [1.0, 1.0, 1.0], "_minimize_overall")