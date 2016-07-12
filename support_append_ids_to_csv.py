# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import zipfile
import numpy as np
from collections import defaultdict
import pickle
import glob
np.random.seed(2016)


def fix_single_file(file_path, type='train'):
    print('Fixing {}...'.format(file_path))
    f = open(file_path)
    if type == 'train':
        pairs = open('../input/ItemPairs_train.csv')
    else:
        pairs = open('../input/ItemPairs_test.csv')
    first_line1 = f.readline()
    first_line2 = pairs.readline()
    if first_line1[:8] == 'itemID_1':
        print('Looks like file already contain IDs {}! Skip it'.format(first_line1))
        f.close()
    dir_upper = os.path.dirname(os.path.dirname(file_path))
    out_file = os.path.join(dir_upper, os.path.basename(file_path) + ".fixed.csv")
    print('Store result in {}...'.format(out_file))
    out = open(out_file, "w")
    out.write("itemID_1,itemID_2," + first_line1)
    count = 0
    while 1:
        line1 = f.readline()
        line2 = pairs.readline()
        if line1 == '':
            if line2 != '':
                print('Incorrect number of lines 1 {}'.format(line2))
            break
        if line2 == '':
            print('Incorrect number of lines')
        arr = line2.split(",")
        if type == 'train':
            out.write(arr[0] + "," + arr[1] + "," + line1)
        else:
            out.write(arr[1] + "," + arr[2].strip() + "," + line1)
        count += 1

    f.close()
    pairs.close()
    out.close()


def fix_ztf(file_path):
    print('Fixing {}...'.format(file_path))
    f = open(file_path)
    pairs = open('../modified_data/ItemPairs_zft.csv')
    first_line1 = f.readline()
    first_line2 = pairs.readline()
    if first_line1[:8] == 'itemID_1':
        print('Looks like file already contain IDs {}! Skip it'.format(first_line1))
        f.close()
    dir_upper = os.path.join(os.path.dirname(os.path.dirname(file_path)), "ztf_fixed")
    out_file = os.path.join(dir_upper, os.path.basename(file_path) + ".fixed.csv")
    print('Store result in {}...'.format(out_file))
    out = open(out_file, "w")
    out.write("itemID_1,itemID_2," + first_line1)
    count = 0
    while 1:
        line1 = f.readline()
        line2 = pairs.readline()
        if line1 == '':
            if line2 != '':
                print('Incorrect number of lines 1 {}'.format(line2))
            break
        if line2 == '':
            print('Incorrect number of lines')
        arr = line2.split(",")
        out.write(arr[0] + "," + arr[1] + "," + line1)
        count += 1

    f.close()
    pairs.close()
    out.close()


def fix_all_in_dir():
    if 0:
        files = glob.glob("../data_external/train/*.csv")
        for f in files:
            fix_single_file(f, 'train')
    if 0:
        files = glob.glob("../data_external/test/*.csv")
        for f in files:
            fix_single_file(f, 'test')
    if 0:
        fix_single_file("../data_external/!new/imagehash_v7_trn.csv", 'train')
        fix_single_file("../data_external/!new/imagehash_v7_tst.csv", 'test')
        fix_single_file("../data_external/!new/word2vec_l5Rus_trn.csv", 'train')
        fix_single_file("../data_external/!new/word2vec_l5Rus_tst.csv", 'test')
        fix_single_file("../data_external/!new/inception_trn.csv", 'train')
        fix_single_file("../data_external/!new/inception_tst.csv", 'test')

    if 0:
        fix_single_file("../data_external/!new/word2vec_l5News_trn.csv", 'train')
        fix_single_file("../data_external/!new/word2vec_l5News_tst.csv", 'test')
        fix_single_file("../data_external/!new/pool3_trn.csv", 'train')
        fix_single_file("../data_external/!new/pool3_tst.csv", 'test')

    if 0:
        fix_single_file("../data_external/!new2/description_trn.csv", 'train')
        fix_single_file("../data_external/!new2/description_tst.csv", 'test')
        fix_single_file("../data_external/!new2/json_m_trn.csv", 'train')
        fix_single_file("../data_external/!new2/json_m_tst.csv", 'test')
        fix_single_file("../data_external/!new2/json_s_trn.csv", 'train')
        fix_single_file("../data_external/!new2/json_s_tst.csv", 'test')
        fix_single_file("../data_external/!new2/json_trn.csv", 'train')
        fix_single_file("../data_external/!new2/json_tst.csv", 'test')
        fix_single_file("../data_external/!new2/sklearn_dw_trn.csv", 'train')
        fix_single_file("../data_external/!new2/sklearn_dw_tst.csv", 'test')
        fix_single_file("../data_external/!new2/sklearn_jw_trn.csv", 'train')
        fix_single_file("../data_external/!new2/sklearn_jw_tst.csv", 'test')
        fix_single_file("../data_external/!new2/sklearn_tdjw_trn.csv", 'train')
        fix_single_file("../data_external/!new2/sklearn_tdjw_tst.csv", 'test')
        fix_single_file("../data_external/!new2/sklearn_tw_trn.csv", 'train')
        fix_single_file("../data_external/!new2/sklearn_tw_tst.csv", 'test')
        fix_single_file("../data_external/!new2/title_trn.csv", 'train')
        fix_single_file("../data_external/!new2/title_tst.csv", 'test')


def fix_all_ztf():
    # fix_ztf("../data_external/!new/imagehash_v7_zft.csv")
    fix_ztf("../data_external/!new/inception_zft.csv")
    fix_ztf("../data_external/!new/pool3_zft.csv")
    fix_ztf("../data_external/!new/word2vec_l5News_zft.csv")
    fix_ztf("../data_external/!new/word2vec_l5Rus_zft.csv")
    fix_ztf("../data_external/!new2/description_m_zft.csv")
    fix_ztf("../data_external/!new2/description_s_zft.csv")
    fix_ztf("../data_external/!new2/description_zft.csv")
    fix_ztf("../data_external/!new2/gensim_zft.csv")
    fix_ztf("../data_external/!new2/gensim_v3_zft.csv")
    fix_ztf("../data_external/!new2/gensim_v4_zft.csv")
    fix_ztf("../data_external/!new2/json_m_zft.csv")
    fix_ztf("../data_external/!new2/json_zft.csv")
    fix_ztf("../data_external/!new2/sklearn_dw_zft.csv")
    fix_ztf("../data_external/!new2/sklearn_jw_zft.csv")
    fix_ztf("../data_external/!new2/sklearn_tdjw_zft.csv")
    fix_ztf("../data_external/!new2/sklearn_tw_zft.csv")
    fix_ztf("../data_external/!new2/title_m_zft.csv")
    fix_ztf("../data_external/!new2/title_s_zft.csv")
    fix_ztf("../data_external/!new2/title_zft.csv")


if __name__ == '__main__':
    # fix_all_in_dir()
    fix_all_ztf()

