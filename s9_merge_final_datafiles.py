# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
import time
import json


# nltk.download('all')
# exit()

def is_same(row, cat1, cat2):
    if row[cat1] == row[cat2]:
        return 1
    return 0


def num_of_images(row):
    if row['images_array'] == -1:
        return 0
    return str(row['images_array']).count(',') + 1


def json_simialrity_type1(row):
    a1 = row['attrsJSON_1']
    a2 = row['attrsJSON_2']
    if a1 == -1 or a2 == -1:
        return 0.0
    j1 = json.loads(str(a1))
    j2 = json.loads(str(a2))
    l1 = len(j1)
    l2 = len(j2)
    if l1 == 0 or l2 == 0:
        return 0.0

    same = 0
    for el in j1:
        if el in j2.keys():
            same += 1

    return 2*same/(l1 + l2)


def json_simialrity_type2(row):
    a1 = row['attrsJSON_1']
    a2 = row['attrsJSON_2']
    if a1 == -1 or a2 == -1:
        return 0.0
    j1 = json.loads(str(a1))
    j2 = json.loads(str(a2))
    l1 = len(j1)
    l2 = len(j2)
    if l1 == 0 or l2 == 0:
        return 0.0

    same = 0
    for el in j1:
        if el in j2.keys():
            if j1[el] == j2[el]:
                same += 1

    return 2*same/(l1 + l2)


def count_proc_json_attr_similar(table, items):
    print("Count json attr similarity parameter...")
    temp1 = items[['itemID', 'attrsJSON']].rename(columns={
         'itemID': 'itemID_1',
         'attrsJSON': 'attrsJSON_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'attrsJSON']].rename(columns={
         'itemID': 'itemID_2',
         'attrsJSON': 'attrsJSON_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table['json_attr_proc_similarity_t1'] = table.apply(json_simialrity_type1, axis=1)
    table['json_attr_proc_similarity_t2'] = table.apply(json_simialrity_type2, axis=1)
    table = table.drop(['attrsJSON_1'], axis=1)
    table = table.drop(['attrsJSON_2'], axis=1)
    print(table['json_attr_proc_similarity_t1'].describe())
    print(table['json_attr_proc_similarity_t2'].describe())
    return table


def get_same_status(pairs, items, target):
    text_compare = pairs
    item1 = items[['itemID', target]]
    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            target: target + '_1',
        }
    )
    text_compare = pd.merge(text_compare, item1, how='left', on='itemID_1', left_index=True)
    item2 = items[['itemID', target]]
    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            target: target + '_2',
        }
    )
    text_compare = pd.merge(text_compare, item2, how='left', on='itemID_2', left_index=True)
    text_compare[target + '_same'] = np.equal(text_compare[target + '_1'], text_compare[target + '_2']).astype(np.int32)
    # print(text_compare[target + '_same'].describe())
    return text_compare[['id', target + '_same']]


def get_str_length(pairs, items, target):
    text_compare = pairs
    item1 = items[['itemID', target]]
    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            target: target + '_1',
        }
    )
    text_compare = pd.merge(text_compare, item1, how='left', on='itemID_1', left_index=True)
    item2 = items[['itemID', target]]
    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            target: target + '_2',
        }
    )
    text_compare = pd.merge(text_compare, item2, how='left', on='itemID_2', left_index=True)
    text_compare[target + '_len_1'] = text_compare[target + '_1'].str.len()
    text_compare[target + '_len_2'] = text_compare[target + '_2'].str.len()
    text_compare[target + '_len_1'].fillna(-1, inplace=True)
    text_compare[target + '_len_2'].fillna(-1, inplace=True)
    text_compare[target + '_len_1'] = text_compare[target + '_len_1'].astype(np.int32)
    text_compare[target + '_len_2'] = text_compare[target + '_len_2'].astype(np.int32)
    # print(text_compare[target + '_same'].describe())
    return text_compare[['id', target + '_len_1', target + '_len_2']]


def count_images_num(table, items, i):
    si = str(i)
    print("Count number of images " + si + "...")
    temp1 = items[['itemID', 'images_array']].rename(columns={
         'itemID': 'itemID_' + si,
    })
    table = pd.merge(table, temp1, how='left', on='itemID_' + si, left_index=True)
    table['images_num_' + si] = table.apply(num_of_images, axis=1)
    table['images_num_' + si] = table['images_num_' + si].astype(np.int32)
    table = table.drop(['images_array'], axis=1)
    print(table['images_num_' + si].describe())
    return table


def get_item_ids_diff(table):
    table['item_id_sub'] = (table['itemID_1'] - table['itemID_2']).abs()
    table['item_id_d1'] = table['itemID_1']/table['itemID_2']
    table['item_id_d2'] = table['itemID_2']/table['itemID_1']
    table['item_id_diff'] = table[['item_id_d1', 'item_id_d2']].max(axis=1)
    table = table.drop(['item_id_d1', 'item_id_d2'], axis=1)
    return table


def decrease_size_dataframe(df):
    float_decr = 0
    int8_decr = 0
    int16_decr = 0
    int32_decr = 0
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
            float_decr += 1
        elif df[col].dtype == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype(np.int8)
                int8_decr += 1
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype(np.int16)
                int16_decr += 1
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
                int32_decr += 1
    # df.to_hdf(f_store, key, format='t', complevel=9, complib='blosc')
    print('Float decrease: {}'.format(float_decr))
    print('Int8 decrease: {}'.format(int8_decr))
    print('Int16 decrease: {}'.format(int16_decr))
    print('Int32 decrease: {}'.format(int32_decr))


def prep_train(testing = 1):

    start_time = time.time()
    input_path = "../input/ItemPairs_train.csv"
    # input_path = "../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv"
    # out_path = "../modified_data/train.csv"
    out_path = "../modified_data/train_original.csv"

    types1 = {
        'itemID_1': np.dtype(np.int32),
        'itemID_2': np.dtype(np.int32),
        'isDuplicate': np.dtype(np.int32),
        'generationMethod': np.dtype(np.int32),
    }

    types2 = {
        'itemID': np.dtype(np.int32),
        'categoryID': np.dtype(np.int32),
        'title': np.dtype(np.str),
        'description': np.dtype(np.str),
        'images_array': np.dtype(np.str),
        'attrsJSON': np.dtype(np.str),
        'price': np.dtype(np.float32),
        'locationID': np.dtype(np.int32),
        'metroID': np.dtype(np.float32),
        'lat': np.dtype(np.float32),
        'lon': np.dtype(np.float32),
    }

    print("Load pairs.csv")
    if testing == 1:
        pairs = pd.read_csv(input_path, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(input_path, dtype=types1)

    # Add 'id' column for easy merge
    pairs['id'] = pairs.index.astype(int)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    location = pd.read_csv("../input/Location.csv")
    category = pd.read_csv("../input/Category.csv")

    train = pairs
    itrain_length = len(train.index)
    print('Initial train length: {}'.format(itrain_length))
    train = train.drop(['generationMethod'], axis=1)

    # It now read at the end of file
    if 0:
        print('Count proc json...')
        train = count_proc_json_attr_similar(train, items)
        save_table = train[['itemID_1', 'itemID_2', 'json_attr_proc_similarity_t1', 'json_attr_proc_similarity_t2']]
        save_table.to_csv("../modified_data/json_info_train.csv", index=False)

    print('Add text features...')
    for f in ['title', 'description', 'attrsJSON']:
        text_length = get_str_length(pairs, items, f)
        train = pd.merge(train, text_length, how='left', on='id', left_index=True)
        # Not needed
        # text_compare = get_same_status(pairs, items, f)
        # train = pd.merge(train, text_compare, how='left', on='id', left_index=True)
        # print(train[f + '_same'].describe())
        # print(train[f + '_len1'].describe())
        # print(train[f + '_len2'].describe())

    print('Merge item 1...')
    item1 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon']]
    item1 = pd.merge(item1, category, how='left', on='categoryID', left_index=True)
    item1 = pd.merge(item1, location, how='left', on='locationID', left_index=True)

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1'
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)
    train = count_images_num(train, items, 1)

    print('Merge item 2...')
    item2 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon']]
    item2 = pd.merge(item2, category, how='left', on='categoryID', left_index=True)
    item2 = pd.merge(item2, location, how='left', on='locationID', left_index=True)

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2'
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)
    train = count_images_num(train, items, 2)

    # Add some more features
    for t in ['price', 'title_len', 'description_len', 'attrsJSON_len']:
        train[t + '_d1'] = train[t + '_1']/train[t + '_2']
        train[t + '_d2'] = train[t + '_2']/train[t + '_1']
        train[t + '_diff'] = train[[t + '_d1', t + '_d2']].max(axis=1)
        train = train.drop([t + '_d1', t + '_d2'], axis=1)
        # print(train[t + '_diff'].describe())
        train[t + '_min'] = train[[t + '_1', t + '_2']].min(axis=1)
        train[t + '_max'] = train[[t + '_1', t + '_2']].max(axis=1)
        # print(train[t + '_min'].describe())
        # print(train[t + '_max'].describe())

    # Price relative diff
    train['price_diff_relative'] = (train['price_1'] - train['price_2']).abs() / (train['price_1'].abs() + train['price_2'].abs())
    print(train['price_diff_relative'].describe())

    train['distance'] = (train['lat_1'] - train['lat_2'])*(train['lat_1'] - train['lat_2']) + \
                        (train['lon_1'] - train['lon_2'])*(train['lon_1'] - train['lon_2'])
    train['distance'] = train['distance'].apply(np.sqrt)
    print(train['distance'].describe())

    # Create same arrays
    print('Create same arrays')
    train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)
    train['locationID_same'] = np.equal(train['locationID_1'], train['locationID_2']).astype(np.int32)
    # They always same
    # train['categoryID_same'] = np.equal(train['categoryID_1'], train['categoryID_2']).astype(np.int32)
    train['regionID_same'] = np.equal(train['regionID_1'], train['regionID_2']).astype(np.int32)
    train['metroID_same'] = np.equal(train['metroID_1'], train['metroID_2']).astype(np.int32)
    train['lat_same'] = np.equal(train['lat_1'], train['lat_2']).astype(np.int32)
    train['lon_same'] = np.equal(train['lon_1'], train['lon_2']).astype(np.int32)
    train['images_num_same'] = np.equal(train['images_num_1'], train['images_num_2']).astype(np.int32)

    # Append haversine distance
    haversine = pd.read_csv('../modified_data/haversine_distance_train.csv')
    train = pd.merge(train, haversine, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['distance_haversine'].describe())

    # Append triples status
    triples_exists = pd.read_csv('../modified_data/triples_exists_train.csv')
    train = pd.merge(train, triples_exists, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['triple_exists'].describe())

    # Append title similarity
    title_similarity = pd.read_csv('../modified_data/titles_info_train.csv')
    print(len(title_similarity.index))
    print(len(train.index))
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_proc_similarity']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    # print(train['title_proc_similarity'].describe())

    # Append descr similarity
    descr_similarity = pd.read_csv('../modified_data/description_info_train.csv')
    train = pd.merge(train, descr_similarity[['itemID_1', 'itemID_2', 'description_proc_similarity']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['description_proc_similarity'].describe())

    # Append title similarity v2
    title_similarity = pd.read_csv('../modified_data/titles_info_train_v2.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_proc_similarity_v2']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['title_proc_similarity_v2'].describe())

    # Append descr similarity v2
    descr_similarity = pd.read_csv('../modified_data/description_info_train_v2.csv')
    train = pd.merge(train, descr_similarity[['itemID_1', 'itemID_2', 'description_proc_similarity_v2']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['description_proc_similarity_v2'].describe())

    if 1:
        # Append json similarity
        json_similarity = pd.read_csv('../modified_data/json_info_train.csv')
        train = pd.merge(train, json_similarity[['itemID_1', 'itemID_2', 'json_attr_proc_similarity_t1', 'json_attr_proc_similarity_t2']], how='left',
                         on=['itemID_1', 'itemID_2'], left_index=True)
        print(len(train.index))
        print(train['json_attr_proc_similarity_t1'].describe())
        print(train['json_attr_proc_similarity_t2'].describe())

    # Append title levinshtein
    title_similarity = pd.read_csv('../modified_data/titles_levinshtein_train.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_proc_levinshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['title_proc_levinshtein'].describe())

    # Append desc levinshtein
    title_similarity = pd.read_csv('../modified_data/descr_levinshtein_train.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'descr_proc_levinshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['descr_proc_levinshtein'].describe())

    # Append title damerau levenshtein
    title_similarity = pd.read_csv('../modified_data/titles_damerau_levenshtein_train.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_proc_damerau_levenshtein_norm', 'title_proc_damerau_levenshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['title_proc_damerau_levenshtein'].describe())
    print(len(train.index))
    print(train['title_proc_damerau_levenshtein_norm'].describe())

    # Append descr damerau levenshtein
    title_similarity = pd.read_csv('../modified_data/descr_damerau_levenshtein_train.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'descr_proc_damerau_levenshtein_norm', 'descr_proc_damerau_levenshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['descr_proc_damerau_levenshtein'].describe())
    print(train['descr_proc_damerau_levenshtein_norm'].describe())

    # Append title jaro_winkler
    title_similarity = pd.read_csv('../modified_data/titles_jaro_winkler_train.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_jaro_winkler_t1', 'title_jaro_winkler_t2']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['title_jaro_winkler_t1'].describe())
    print(train['title_jaro_winkler_t2'].describe())

    # Append descr jaro_winkler
    title_similarity = pd.read_csv('../modified_data/descr_jaro_winkler_train.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'descr_jaro_winkler_t1', 'descr_jaro_winkler_t2']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['descr_jaro_winkler_t1'].describe())
    print(train['descr_jaro_winkler_t2'].describe())

    # Append image params
    image_params = pd.read_csv('../modified_data/images_params_train.csv')
    train = pd.merge(train, image_params[['itemID_1', 'itemID_2', 'have_same', 'min_diff_hash1', 'min_diff_hash2', 'min_diff_hash3']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(len(train.index))
    print(train['have_same'].describe())
    print(train['min_diff_hash1'].describe())
    print(train['min_diff_hash2'].describe())
    print(train['min_diff_hash3'].describe())
    print(train['have_same'].isnull().sum())

    # Difference between item IDs
    train = get_item_ids_diff(train)

    print('Drop id column...')
    train = train.drop(['id'], axis=1)

    # Append json additional tables
    print('Append JSON additional tables...')
    json_table2 = pd.read_csv('../modified_data/json_text_sim_params_train.csv')
    print('Nans:', json_table2.isnull().sum().sum())
    train = pd.merge(train, json_table2, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print('Nans:', train.isnull().sum().sum())
    print(len(train.index))

    if 1:
        json_table1 = pd.read_csv('../modified_data/json_same_params_train.csv')
        print('Nans:', json_table1.isnull().sum().sum())
        print(json_table1.describe())
        train = pd.merge(train, json_table1, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
        print(len(train.index))

    # Additional data GEN START
    add_table = pd.read_csv('../data_external/cat_trn.csv.fixed.csv')
    # print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/description_m_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/description_s_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/gensim_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/gensim_v3_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/gensim_v4_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/imagehash_v7_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/pcat_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/title_m_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/title_s_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/word2vec_l5News_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/word2vec_l5Rus_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/inception_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/pool3_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/description_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/json_m_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/json_s_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/json_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_dw_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_jw_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_tdjw_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_tw_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/title_trn.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    # Additional data GEN END


    add_table = pd.read_csv('../modified_data/train_IDs_features.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    if (itrain_length != len(train.index)):
        print('There was some error with LENGTH!!!!! Check it! {} != {}'.format(itrain_length, len(train.index)))

    print('Number of features: {}'.format(len(train.columns.values)))

    # print(train.describe())
    if testing == 0:
        print('Saving train...')
        # train.to_csv(out_path, index=False)
        decrease_size_dataframe(train)
        train.to_hdf(out_path + '.hdf', 'table', format='t', complevel=9, complib='blosc')
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def print_nan_stat(table):
    for f in list(table.columns.values):
        if table[f].isnull().any():
            print(f, " ", table[f].isnull().sum().sum())


def prep_test(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'id': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_test.csv")
    if testing == 1:
        pairs = pd.read_csv("../input/ItemPairs_test.csv", dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv("../input/ItemPairs_test.csv", dtype=types1)
    print("Load ItemInfo_test.csv")
    items = pd.read_csv("../input/ItemInfo_test.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    location = pd.read_csv("../input/Location.csv")
    category = pd.read_csv("../input/Category.csv")

    train = pairs
    itrain_length = len(train.index)

    # It now read at the end of file
    if 0:
        train = count_proc_json_attr_similar(train, items)
        save_table = train[['id', 'json_attr_proc_similarity_t1', 'json_attr_proc_similarity_t2']]
        save_table.to_csv("../modified_data/json_info_test.csv", index=False)


    print('Add text features...')
    for f in ['title', 'description', 'attrsJSON']:
        text_length = get_str_length(pairs, items, f)
        train = pd.merge(train, text_length, how='left', on='id', left_index=True)
        # Not needed
        # text_compare = get_same_status(pairs, items, f)
        # train = pd.merge(train, text_compare, how='left', on='id', left_index=True)
        # print(train[f + '_same'].describe())
        # print(train[f + '_len1'].describe())
        # print(train[f + '_len2'].describe())

    print('Merge item 1...')
    item1 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon']]
    item1 = pd.merge(item1, category, how='left', on='categoryID', left_index=True)
    item1 = pd.merge(item1, location, how='left', on='locationID', left_index=True)

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1'
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)
    train = count_images_num(train, items, 1)

    print('Merge item 2...')
    item2 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon']]
    item2 = pd.merge(item2, category, how='left', on='categoryID', left_index=True)
    item2 = pd.merge(item2, location, how='left', on='locationID', left_index=True)

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2'
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)
    train = count_images_num(train, items, 2)

    # Add some more features
    for t in ['price', 'title_len', 'description_len', 'attrsJSON_len']:
        train[t + '_d1'] = train[t + '_1']/train[t + '_2']
        train[t + '_d2'] = train[t + '_2']/train[t + '_1']
        train[t + '_diff'] = train[[t + '_d1', t + '_d2']].max(axis=1)
        train = train.drop([t + '_d1', t + '_d2'], axis=1)
        print(train[t + '_diff'].describe())
        train[t + '_min'] = train[[t + '_1', t + '_2']].min(axis=1)
        train[t + '_max'] = train[[t + '_1', t + '_2']].max(axis=1)
        print(train[t + '_min'].describe())
        print(train[t + '_max'].describe())

    # Price relative diff
    train['price_diff_relative'] = (train['price_1'] - train['price_2']).abs() / (train['price_1'].abs() + train['price_2'].abs())
    print(train['price_diff_relative'].describe())

    train['distance'] = (train['lat_1'] - train['lat_2'])*(train['lat_1'] - train['lat_2']) + \
                        (train['lon_1'] - train['lon_2'])*(train['lon_1'] - train['lon_2'])
    train['distance'] = train['distance'].apply(np.sqrt)
    print(train['distance'].describe())

    # titleStringDist = stringdist(title_1, title_2, method = "jw"),
    # titleStringDist2 = (stringdist(title_1, title_2, method = "lcs") /
    # pmax(ncharTitle_1, ncharTitle_2, na.rm=TRUE)),

    # Create same arrays
    print('Create same arrays')
    train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)
    train['locationID_same'] = np.equal(train['locationID_1'], train['locationID_2']).astype(np.int32)
    # They always same
    # train['categoryID_same'] = np.equal(train['categoryID_1'], train['categoryID_2']).astype(np.int32)
    train['regionID_same'] = np.equal(train['regionID_1'], train['regionID_2']).astype(np.int32)
    train['metroID_same'] = np.equal(train['metroID_1'], train['metroID_2']).astype(np.int32)
    train['lat_same'] = np.equal(train['lat_1'], train['lat_2']).astype(np.int32)
    train['lon_same'] = np.equal(train['lon_1'], train['lon_2']).astype(np.int32)
    train['images_num_same'] = np.equal(train['images_num_1'], train['images_num_2']).astype(np.int32)

    # Append haversine distance
    haversine = pd.read_csv('../modified_data/haversine_distance_test.csv')
    train = pd.merge(train, haversine, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['distance_haversine'].describe())

    # Append triples status
    triples_exists = pd.read_csv('../modified_data/triples_exists_test.csv')
    train = pd.merge(train, triples_exists, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['triple_exists'].describe())

    # Append title similarity
    title_similarity = pd.read_csv('../modified_data/titles_info_test.csv')
    train = pd.merge(train, title_similarity[['id', 'title_proc_similarity']], how='left', on=['id'], left_index=True)
    # print(train['title_proc_similarity'].describe())

    # Append descr similarity
    descr_similarity = pd.read_csv('../modified_data/description_info_test.csv')
    train = pd.merge(train, descr_similarity[['id', 'description_proc_similarity']], how='left', on=['id'], left_index=True)
    # print(train['description_proc_similarity'].describe())

    # Append title similarity v2
    title_similarity = pd.read_csv('../modified_data/titles_info_test_v2.csv')
    train = pd.merge(train, title_similarity[['id', 'title_proc_similarity_v2']], how='left', on=['id'], left_index=True)
    # print(train['title_proc_similarity'].describe())

    # Append descr similarity v2
    descr_similarity = pd.read_csv('../modified_data/description_info_test_v2.csv')
    train = pd.merge(train, descr_similarity[['id', 'description_proc_similarity_v2']], how='left', on=['id'], left_index=True)
    # print(train['description_proc_similarity'].describe())

    # Append json similarity
    descr_similarity = pd.read_csv('../modified_data/json_info_test.csv')
    train = pd.merge(train, descr_similarity[['id', 'json_attr_proc_similarity_t1', 'json_attr_proc_similarity_t2']],
                     how='left', on=['id'], left_index=True)
    # print(train['json_attr_proc_similarity_t1'].describe())
    # print(train['json_attr_proc_similarity_t2'].describe())

    # Append title levenshtein
    title_similarity = pd.read_csv('../modified_data/titles_levinshtein_test.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_proc_levinshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['title_proc_levinshtein'].describe())

    # Append desc levenshtein
    title_similarity = pd.read_csv('../modified_data/descr_levinshtein_test.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'descr_proc_levinshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['descr_proc_levinshtein'].describe())

    # Append title damerau levenshtein
    title_similarity = pd.read_csv('../modified_data/titles_damerau_levenshtein_test.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_proc_damerau_levenshtein_norm', 'title_proc_damerau_levenshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['title_proc_damerau_levenshtein'].describe())
    print(train['title_proc_damerau_levenshtein_norm'].describe())

    # Append descr damerau levenshtein
    title_similarity = pd.read_csv('../modified_data/descr_damerau_levenshtein_test.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'descr_proc_damerau_levenshtein_norm', 'descr_proc_damerau_levenshtein']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['descr_proc_damerau_levenshtein'].describe())
    print(train['descr_proc_damerau_levenshtein_norm'].describe())

    # Append title jaro_winkler
    title_similarity = pd.read_csv('../modified_data/titles_jaro_winkler_test.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'title_jaro_winkler_t1', 'title_jaro_winkler_t2']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['title_jaro_winkler_t1'].describe())
    print(train['title_jaro_winkler_t2'].describe())

    # Append descr jaro_winkler
    title_similarity = pd.read_csv('../modified_data/descr_jaro_winkler_test.csv')
    train = pd.merge(train, title_similarity[['itemID_1', 'itemID_2', 'descr_jaro_winkler_t1', 'descr_jaro_winkler_t2']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['descr_jaro_winkler_t1'].describe())
    print(train['descr_jaro_winkler_t2'].describe())

    # Append image params
    image_params = pd.read_csv('../modified_data/images_params_test.csv')
    train = pd.merge(train, image_params[['itemID_1', 'itemID_2', 'have_same', 'min_diff_hash1', 'min_diff_hash2', 'min_diff_hash3']], how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(train['have_same'].describe())
    print(train['min_diff_hash1'].describe())
    print(train['min_diff_hash2'].describe())
    print(train['min_diff_hash3'].describe())
    print(train['have_same'].isnull().sum())

    # Difference between item IDs
    train = get_item_ids_diff(train)

    # Append json additional tables
    print('Append JSON additional tables...')
    json_table2 = pd.read_csv('../modified_data/json_text_sim_params_test.csv')
    print('Nans:', json_table2.isnull().sum().sum())
    train = pd.merge(train, json_table2, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(json_table2.describe())

    json_table1 = pd.read_csv('../modified_data/json_same_params_test.csv')
    train = pd.merge(train, json_table1, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    print(json_table1.describe())

    # Additional data GEN START
    add_table = pd.read_csv('../data_external/cat_tst.csv.fixed.csv')
    # print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/description_m_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/description_s_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/gensim_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/gensim_v3_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/gensim_v4_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    if 0:
        add_table = pd.read_csv('../data_external/!old/imagehash_v3_tst.csv.fixed.csv')
        print(add_table.describe())
        train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

        add_table = pd.read_csv('../data_external/!old/imagehash_v5_tst.csv.fixed.csv')
        print(add_table.describe())
        train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

        add_table = pd.read_csv('../data_external/!old/imagehash_v6_tst.csv.fixed.csv')
        print(add_table.describe())
        train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    if 1:
        add_table = pd.read_csv('../data_external/imagehash_v7_tst.csv.fixed.csv')
        print(add_table.describe())
        train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/pcat_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/title_m_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/title_s_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/word2vec_l5News_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/word2vec_l5Rus_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/inception_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/pool3_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/description_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/json_m_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/json_s_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/json_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_dw_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_jw_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_tdjw_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/sklearn_tw_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    add_table = pd.read_csv('../data_external/title_tst.csv.fixed.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
    # Additional data GEN END

    add_table = pd.read_csv('../modified_data/test_IDs_features.csv')
    print(add_table.describe())
    train = pd.merge(train, add_table, how='left', on=['itemID_1', 'itemID_2'], left_index=True)

    if (itrain_length != len(train.index)):
        print('There was some error with LENGTH!!!!! Check it! {} != {}'.format(itrain_length, len(train.index)))

    nan_number = train.isnull().sum().sum()
    if nan_number > 0:
        print('Warnings Nans in output table! Check it. Number of nans:', nan_number)
        print(train.isnull().any(axis=0))
        print_nan_stat(train)

    print('Number of features: {}'.format(len(train.columns.values)))

    # print(train.describe())
    if testing == 0:
        print('Saving test...')
        # train.to_csv("../modified_data/test.csv", index=False)
        decrease_size_dataframe(train)
        train.to_hdf('../modified_data/test.hdf', 'table', format='t', complevel=9, complib='blosc')
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))

prep_train(0)
# prep_test(0)
