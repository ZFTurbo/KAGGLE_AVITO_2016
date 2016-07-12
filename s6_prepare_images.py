# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import defaultdict
import time
import copy
import json
import glob
import hashlib
import os
import pickle
import imagehash
from PIL import Image
import json
import math

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


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def getImageHash(img):
    io = Image.open(img)
    hash1 = imagehash.average_hash(io)
    hash2 = imagehash.phash(io)
    hash3 = imagehash.dhash(io)
    return hash1, hash2, hash3

'''
def hex_to_hash_my(hexstr):
    l = []
    for i in range(len(hexstr) >> 1):
        h = hexstr[i*2:i*2+2]
        v = int("0x" + h, 16)
        l.append([v & 2**i > 0 for i in range(8)])
    return imagehash.ImageHash(np.array(l))
'''

def get_md5_all_images(folder):
    cache_file_paths = "../modified_data/im/all_images_" + str(folder) + ".dat"
    cahche_image_data = "../modified_data/im/image_hashes_" + str(folder) + ".dat"
    if not os.path.isfile(cache_file_paths):
        files = glob.glob("../input/Images_" + str(folder) + "/*/*.jpg")
        cache_data(files, cache_file_paths)
    else:
        files = restore_data(cache_file_paths)

    if not os.path.isfile(cahche_image_data):
        img_storage = dict()
    else:
        img_storage = restore_data(cahche_image_data)

    total = 0
    print(len(files))
    for fl in files:
        total += 1
        name = os.path.basename(fl)
        id = name[:-4]
        if id in img_storage:
            continue

        if not os.path.isfile(fl):
            continue

        if os.path.getsize(fl) == 0:
            continue

        if total % 300000 == 0:
            print('Saving current data [{}]...'.format(total))
            cache_data(img_storage, cahche_image_data)

        md = md5(fl)
        h1, h2, h3 = getImageHash(fl)
        img_storage[id] = (md, h1, h2, h3)

    cache_data(img_storage, cahche_image_data)


def store_all_images_textinfo(replace = 0):
    for i in range(0, 10):
        print('Reading from cache: {}'.format(i))
        cahche_image_data = "../modified_data/im/image_hashes_" + str(i) + ".dat"
        storage = restore_data(cahche_image_data)
        for id in storage:
            if len(id) <= 1:
                folder = int(id[-1:])
            else:
                folder = int(id[-2:])
            dir_name = os.path.join("..", "modified_data", "imdata", str(folder))
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            path = os.path.join(dir_name, str(id) + ".txt")
            if not os.path.isfile(path) or replace == 1:
                # print(path, id, folder, storage[id])
                json.dump((storage[id][0], str(storage[id][1]), str(storage[id][2]), str(storage[id][3])), open(path, 'w'))


def get_image_params(im_str, im_full_arr):
    if len(im_str) > 2:
        suffix = int(im_str[-2:])
    else:
        suffix = int(im_str)
    '''
    path = os.path.join('..', 'modified_data', 'imdata', str(suffix), im_str + '.txt')
    if os.path.isfile(path):
        try:
            params = json.load(open(path, 'r'))
        except:
            print('Error [Type 1]: ', im_str)
            params = []
    else:
        print('Error [No file]', im_str)
        params = []
    '''
    if im_str in im_full_arr:
        params = im_full_arr[im_str]
    else:
        print('Error [No file]', im_str)
        params = []
    return params


def find_images_params(ia1, ia2, im_full_arr):
    # print(ia1, ia2)
    arr1 = []
    for i in range(len(ia1)):
        arr1.append(get_image_params(ia1[i], im_full_arr))
    arr2 = []
    for i in range(len(ia2)):
        arr2.append(get_image_params(ia2[i], im_full_arr))

    # print(arr1)
    # print(arr2)

    have_same = 0
    min_diff_hash1 = 99999999999
    min_diff_hash2 = 99999999999
    min_diff_hash3 = 99999999999
    for i in range(len(ia1)):
        param1 = arr1[i]
        if len(param1) == 0:
            continue
        for j in range(len(ia2)):
            param2 = arr2[j]
            if len(param2) == 0:
                continue
            if param1[0] == param2[0]:
                have_same += 1
            h1 = imagehash.hex_to_hash(param1[1]) - imagehash.hex_to_hash(param2[1])
            h2 = imagehash.hex_to_hash(param1[2]) - imagehash.hex_to_hash(param2[2])
            h3 = imagehash.hex_to_hash(param1[3]) - imagehash.hex_to_hash(param2[3])
            if h1 < min_diff_hash1:
                min_diff_hash1 = h1
            if h2 < min_diff_hash2:
                min_diff_hash2 = h2
            if h3 < min_diff_hash3:
                min_diff_hash3 = h3
            # print(have_same, h1, h2, h3)

    return have_same, min_diff_hash1, min_diff_hash2, min_diff_hash3


def proc_images_axis(row, im_full_arr):
    answ = pd.Series({'itemID_1': row['itemID_1'], 'itemID_2': row['itemID_2'], 'have_same': -1, 'min_diff_hash1': -1, 'min_diff_hash2': -1, 'min_diff_hash3': -1})
    im1 = str(row['images_array_1'])
    im2 = str(row['images_array_2'])
    if im1 != 'nan' and im2 != 'nan':
        # print(im1)
        # print(im2)
        im_arr1 = im1.split(',')
        im_arr2 = im2.split(',')
        for i in range(len(im_arr1)):
            im_arr1[i] = im_arr1[i].strip()
        for i in range(len(im_arr2)):
            im_arr2[i] = im_arr2[i].strip()
        # print('Duplicate status: ', row['isDuplicate'])
        have_same, min_diff_hash1, min_diff_hash2, min_diff_hash3 = find_images_params(im_arr1, im_arr2, im_full_arr)
        answ['have_same'] = have_same
        answ['min_diff_hash1'] = min_diff_hash1
        answ['min_diff_hash2'] = min_diff_hash2
        answ['min_diff_hash3'] = min_diff_hash3
    return answ


def process_images_train(training, im_array_path):
    cache_path1 = '../modified_data/cache/train_small_cache.pickle.dat'
    if os.path.isfile(cache_path1) and training == 1:
        train = restore_data(cache_path1)
    else:
        # ipairs = "../input/ItemPairs_train.csv"
        ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
        if training == 1:
            pairs = pd.read_csv(ipairs)[0:1000]
        else:
            pairs = pd.read_csv(ipairs)
        items = pd.read_csv("../input/ItemInfo_train.csv")
        train = pairs

        print('Merge item 1...')
        item1 = items[['itemID', 'images_array']]
        item1 = item1.rename(
            columns={
                'itemID': 'itemID_1',
                'images_array': 'images_array_1',
            }
        )
        # Add item 1 data
        train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)
        print('Merge item 2...')
        item1 = items[['itemID', 'images_array']]
        item1 = item1.rename(
            columns={
                'itemID': 'itemID_2',
                'images_array': 'images_array_2',
            }
        )
        # Add item 2 data
        train = pd.merge(train, item1, how='left', on='itemID_2', left_index=True)
        if training == 1:
            cache_data(train, cache_path1)

    print('Restore images array...')
    im_arr = restore_data(im_array_path)
    # Split on 30 parts
    batch = 30
    for i in range(1, batch):
        rng = math.floor(len(train.index)/batch)
        start_time = time.time()
        start = i*rng
        end = (i+1)*rng
        if i == batch - 1:
            train1 = train[start:].copy()
        else:
            train1 = train[start:end].copy()
        print('Find images params [{} ({} - {})]...'.format(i, start, end))
        ret = train1.apply(proc_images_axis, args=(im_arr,), axis=1)
        train1 = pd.merge(train1, ret, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
        train1 = train1.drop(['images_array_1'], axis=1)
        train1 = train1.drop(['images_array_2'], axis=1)
        if training != 1:
            train1.to_csv('../modified_data/imcsv/images_params_train_{}.csv'.format(i), index=False)
        print('Create train images param time: {} seconds'.format(round(time.time() - start_time, 2)))


def process_images_test(training, im_array_path):
    cache_path1 = '../modified_data/cache/test_small_cache.pickle.dat'
    if os.path.isfile(cache_path1) and training == 1:
        train = restore_data(cache_path1)
    else:
        if training == 1:
            pairs = pd.read_csv("../input/ItemPairs_test.csv")[0:1000]
        else:
            pairs = pd.read_csv("../input/ItemPairs_test.csv")
        items = pd.read_csv("../input/ItemInfo_test.csv")
        train = pairs

        print('Merge item 1...')
        item1 = items[['itemID', 'images_array']]
        item1 = item1.rename(
            columns={
                'itemID': 'itemID_1',
                'images_array': 'images_array_1',
            }
        )
        # Add item 1 data
        train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)
        print('Merge item 2...')
        item1 = items[['itemID', 'images_array']]
        item1 = item1.rename(
            columns={
                'itemID': 'itemID_2',
                'images_array': 'images_array_2',
            }
        )
        # Add item 2 data
        train = pd.merge(train, item1, how='left', on='itemID_2', left_index=True)
        if training == 1:
            cache_data(train, cache_path1)

    print('Restore images array...')
    im_arr = restore_data(im_array_path)
    # Split on 30 parts
    batch = 30
    for i in range(0, batch):
        rng = math.floor(len(train.index)/batch)
        start_time = time.time()
        start = i*rng
        end = (i+1)*rng
        if i == batch - 1:
            train1 = train[start:].copy()
        else:
            train1 = train[start:end].copy()
        print('Find images params [{} ({} - {})]...'.format(i, start, end))
        ret = train1.apply(proc_images_axis, args=(im_arr,), axis=1)
        train1 = pd.merge(train1, ret, how='left', on=['itemID_1', 'itemID_2'], left_index=True)
        train1 = train1.drop(['images_array_1'], axis=1)
        train1 = train1.drop(['images_array_2'], axis=1)
        if training != 1:
            train1.to_csv('../modified_data/imcsv/images_params_test_{}.csv'.format(i), index=False)
        print('Create test images param time: {} seconds'.format(round(time.time() - start_time, 2)))


def create_image_data_array(out_path):
    start_time = time.time()
    for folder in range(0, 100):
        im_arr = dict()
        folder_out = out_path + '_' + str(folder) + '.pickle.dat'
        if os.path.isfile(folder_out):
            print('File for folder {} already exists. Skipping!'.format(folder))
            continue
        print('Start folder:', folder)
        dir_name = os.path.join("..", "modified_data", "imdata", str(folder), "*.txt")
        files = glob.glob(dir_name)
        total = 0
        print('Number of files:', len(files))
        for fl in files:
            total += 1
            if (total%10000 == 0):
                print('Proccessed {} files...'.format(total))
            index = os.path.basename(fl)[:-4]
            handle = open(fl, 'r')
            try:
                params = json.load(handle)
            except:
                print(fl)
                exit()
            handle.close()
            im_arr[index] = params

        cache_data(im_arr, folder_out)

    print('Elapsed time: {} seconds'.format(round(time.time() - start_time, 2)))


def merge_image_data(in_path):
    overall_dict = dict()
    for folder in range(0, 100):
        folder_out = in_path + '_' + str(folder) + '.pickle.dat'
        d1 = restore_data(folder_out)
        print('Merge {}. Length: {}'.format(folder, len(d1)))
        overall_dict = merge_two_dicts(overall_dict, d1)

    cache_data(overall_dict, '../modified_data/image_params_array.overall.pickle.dat')


if 0:
    for i in range(10):
        get_md5_all_images(i)


def merge_csvs_train_and_test():
    out = open("../modified_data/images_params_train.csv", "w")
    files = glob.glob("../modified_data/imcsv/images_params_train_*.csv")
    header = 0
    for fl in files:
        f = open(fl)
        line = f.readline()
        if header == 0:
            out.write(line)
            header = 1
        while 1:
            line = f.readline()
            if line == '':
                break
            out.write(line)
        f.close()
    out.close()

    out = open("../modified_data/images_params_test.csv", "w")
    files = glob.glob("../modified_data/imcsv/images_params_test_*.csv")
    header = 0
    for fl in files:
        f = open(fl)
        line = f.readline()
        if header == 0:
            out.write(line)
            header = 1
        while 1:
            line = f.readline()
            if line == '':
                break
            out.write(line)
        f.close()
    out.close()

# store_all_images_textinfo(0)
# create_image_data_array('../modified_data/image_params_array')
# merge_image_data('../modified_data/image_params_array')
process_images_train(0, '../modified_data/image_params_array.overall.pickle.dat')
# process_images_test(0, '../modified_data/image_params_array.overall.pickle.dat')
merge_csvs_train_and_test()


'''
Error [Type 1]:  189842
Error [No file] 9322814
Error [No file] 9126754
Error [No file] 651267
Error [No file] 11775274
Error [No file] 10684636
Error [No file] 13949795
Error [No file] 14366881
Error [No file] 13717141
Error [No file] 12961761
Error [No file] 9943584
Error [No file] 12953041
Error [No file] 4515613
Error [No file] 13717141
'''