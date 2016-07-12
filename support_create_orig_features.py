# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
import time
import json


def extract_from_train():
    start_time = time.time()
    input_path = "../modified_data/train_temp.csv"

    print("Load train.csv")
    train = pd.read_csv(input_path)

    print('Write title features...')
    train[['itemID_1',
           'itemID_2',
           'title_proc_similarity',
           'title_proc_similarity_v2',
           'title_proc_levinshtein',
           'title_proc_damerau_levenshtein',
           'title_proc_damerau_levenshtein_norm',
           'title_jaro_winkler_t1',
           'title_jaro_winkler_t2',
           'title_len_diff',
           'title_len_1',
           'title_len_2',
           'title_len_max',
           'title_len_min',
           ]].to_csv("../orig_features/train_titles_param.csv", index=False)

    print('Write descr features...')
    train[['itemID_1',
           'itemID_2',
           'description_proc_similarity',
           'description_proc_similarity_v2',
           'descr_proc_levinshtein',
           'descr_proc_damerau_levenshtein',
           'descr_proc_damerau_levenshtein_norm',
           'descr_jaro_winkler_t1',
           'descr_jaro_winkler_t2',
           'description_len_diff',
           'description_len_1',
           'description_len_2',
           'description_len_max',
           'description_len_min',
           ]].to_csv("../orig_features/train_descr_param.csv", index=False)

    print('Write Images features...')
    train[['itemID_1',
           'itemID_2',
           'images_num_1',
           'images_num_2',
           'images_num_same',
           'have_same',
           'min_diff_hash1',
           'min_diff_hash2',
           'min_diff_hash3',
           ]].to_csv("../orig_features/train_image_param.csv", index=False)

    print('Write JSON features...')
    train[['itemID_1',
           'itemID_2',
           'attrsJSON_len_1',
           'attrsJSON_len_2',
           'attrsJSON_len_diff',
           'attrsJSON_len_max',
           'attrsJSON_len_min',
           'json_attr_proc_similarity_t1',
           'json_attr_proc_similarity_t2',
           ]].to_csv("../orig_features/train_json_simple_param.csv", index=False)

    print('Write MISC features...')
    train[['itemID_1',
           'itemID_2',
           'distance',
           'item_id_diff',
           'item_id_sub',
           'price_1',
           'price_2',
           'price_diff',
           'price_max',
           'price_min',
           'price_same',
           ]].to_csv("../orig_features/train_misc_param.csv", index=False)

    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def extract_from_test():
    start_time = time.time()
    input_path = "../modified_data/test.csv"

    print("Load test.csv")
    train = pd.read_csv(input_path)

    print('Write title features...')
    train[['itemID_1',
           'itemID_2',
           'title_proc_similarity',
           'title_proc_similarity_v2',
           'title_proc_levinshtein',
           'title_proc_damerau_levenshtein',
           'title_proc_damerau_levenshtein_norm',
           'title_jaro_winkler_t1',
           'title_jaro_winkler_t2',
           'title_len_diff',
           'title_len_1',
           'title_len_2',
           'title_len_max',
           'title_len_min',
           ]].to_csv("../orig_features/test_titles_param.csv", index=False)

    print('Write descr features...')
    train[['itemID_1',
           'itemID_2',
           'description_proc_similarity',
           'description_proc_similarity_v2',
           'descr_proc_levinshtein',
           'descr_proc_damerau_levenshtein',
           'descr_proc_damerau_levenshtein_norm',
           'descr_jaro_winkler_t1',
           'descr_jaro_winkler_t2',
           'description_len_diff',
           'description_len_1',
           'description_len_2',
           'description_len_max',
           'description_len_min',
           ]].to_csv("../orig_features/test_descr_param.csv", index=False)

    print('Write Images features...')
    train[['itemID_1',
           'itemID_2',
           'images_num_1',
           'images_num_2',
           'images_num_same',
           'have_same',
           'min_diff_hash1',
           'min_diff_hash2',
           'min_diff_hash3',
           ]].to_csv("../orig_features/test_image_param.csv", index=False)

    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    print('Write JSON features...')
    train[['itemID_1',
           'itemID_2',
           'attrsJSON_len_1',
           'attrsJSON_len_2',
           'attrsJSON_len_diff',
           'attrsJSON_len_max',
           'attrsJSON_len_min',
           'json_attr_proc_similarity_t1',
           'json_attr_proc_similarity_t2',
           ]].to_csv("../orig_features/test_json_simple_param.csv", index=False)

    print('Write MISC features...')
    train[['itemID_1',
           'itemID_2',
           'distance',
           'item_id_diff',
           'item_id_sub',
           'price_1',
           'price_2',
           'price_diff',
           'price_max',
           'price_min',
           'price_same',
           ]].to_csv("../orig_features/test_misc_param.csv", index=False)

    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


extract_from_train()
# extract_from_test()




