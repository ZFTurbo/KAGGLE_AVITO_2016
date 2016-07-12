# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import defaultdict
import time
import copy
import json
import jellyfish
import nltk, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance, damerau_levenshtein_distance_withNPArray, normalized_damerau_levenshtein_distance_withNPArray


stemmer = nltk.stem.snowball.SnowballStemmer('russian')
remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)
stp_wrds = stopwords.words('russian')


def stem_tokens(tokens):
    global stp_wrds
    lst = [stemmer.stem(item) for item in tokens]
    filtered_words = [word for word in lst if word not in stp_wrds]
    return filtered_words


'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


'''remove punctuation, lowercase, stem'''
def normalize_v2(text):
    return nltk.word_tokenize(text.lower().translate(remove_punctuation_map))


def prepareVectorizer():
    vectorizer1 = TfidfVectorizer(tokenizer=normalize, lowercase=True)
    return vectorizer1


def prepareVectorizer_v2():
    vectorizer2 = TfidfVectorizer(tokenizer=normalize_v2, lowercase=True)
    return vectorizer2


def cosine_sim(text1, text2, vectorizer):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except:
        return 0.0
    return ((tfidf * tfidf.T).A)[0, 1]


def test_vectorizer():
    vect = prepareVectorizer()
    text1 = 'Телефон в хорошем состоянии, трещин и сколов нет, за все время менялся только аккумулятор(поэтому заряд держит хорошо), остальное все родное, в целом работает отлично! В комплекте кабель. Обмен не интересен.'
    text2 = 'Продам телефон в хорошем состоянии Полностью рабочий есть WiFi'
    # text1 = 'в б'
    # text2 = 'б'
    print(normalize(text1))
    print(normalize(text2))
    print(cosine_sim(text1, text2, vect))
    exit()


def test_jellyfish():
    text1 = 'Телефон в хорошем состоянии, трещин и сколов нет, за все время менялся только аккумулятор(поэтому заряд держит хорошо), остальное все родное, в целом работает отлично! В комплекте кабель. Обмен не интересен.'
    text2 = 'Продам телефон в хорошем состоянии Полностью рабочий есть WiFi'
    lst1 = normalize(text1)
    lst2 = normalize(text2)
    text_norm1 = ' '.join(lst1)
    text_norm2 = ' '.join(lst2)
    print(jellyfish.jaro_distance(text1, text2))
    print(jellyfish.jaro_distance(text_norm1, text_norm2))
    print(jellyfish.jaro_winkler(text1, text2))
    print(jellyfish.jaro_winkler(text_norm1, text_norm2))
    print(jellyfish.nysiis(text1))
    print(jellyfish.nysiis(text2))
    exit()


def levenshtein_numpy(source, target):
    try:
        if len(source) < len(target):
            return levenshtein_numpy(target, source)
    except:
        print('Error', target, source)
        return 0

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


# test_vectorizer()
# exit()

counter_d = 0


def title_similarity(row, vectorizer):
    if row['title_1'] == row['title_2']:
        return 1.0
    return cosine_sim(row['title_1'], row['title_2'], vectorizer)


def title_levinshtein(row):
    return levenshtein_numpy(row['title_1'], row['title_2'])


def title_damerau_levenshtein_norm(row):
    return normalized_damerau_levenshtein_distance(row['title_1'], row['title_2'])


def title_damerau_levenshtein(row):
    return damerau_levenshtein_distance(row['title_1'], row['title_2'])


def title_jaro_winkler_t1(row):
    return jellyfish.jaro_winkler(row['title_1'], row['title_2'])


def title_jaro_winkler_t2(row):
    lst1 = normalize(row['title_1'])
    lst2 = normalize(row['title_2'])
    text_norm1 = ' '.join(lst1)
    text_norm2 = ' '.join(lst2)
    return jellyfish.jaro_winkler(text_norm1, text_norm2)


def descr_levinshtein(row):
    return levenshtein_numpy(row['description_1'], row['description_2'])


def descr_damerau_levenshtein_norm(row):
    return normalized_damerau_levenshtein_distance(row['description_1'], row['description_2'])


def descr_damerau_levenshtein(row):
    return damerau_levenshtein_distance(row['description_1'], row['description_2'])


def descr_jaro_winkler_t1(row):
    return jellyfish.jaro_winkler(row['description_1'], row['description_2'])


def descr_jaro_winkler_t2(row):
    lst1 = normalize(row['description_1'])
    lst2 = normalize(row['description_2'])
    text_norm1 = ' '.join(lst1)
    text_norm2 = ' '.join(lst2)
    return jellyfish.jaro_winkler(text_norm1, text_norm2)


def description_similarity(row, vectorizer):
    global counter_d
    if row['description_1'] == row['description_2']:
        return 1.0
    if row['description_1'] == '':
        return 0.0
    if row['description_2'] == '':
        return 0.0
    counter_d += 1
    if counter_d % 100000 == 0:
        print('Proccessed ' + str(counter_d) + ' rows...')

    return cosine_sim(row['description_1'], row['description_2'], vectorizer)


def count_proc_title_similar(table, items):
    print("Count title similarity parameter...")
    temp1 = items[['itemID', 'title']].rename(columns={
         'itemID': 'itemID_1',
         'title': 'title_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'title']].rename(columns={
         'itemID': 'itemID_2',
         'title': 'title_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    vectorizer = prepareVectorizer()
    table['title_proc_similarity'] = table.apply(title_similarity, args=(vectorizer,), axis=1)
    table = table.drop(['title_1'], axis=1)
    table = table.drop(['title_2'], axis=1)
    print(table['title_proc_similarity'].describe())
    return table


def count_proc_title_similar_v2(table, items):
    print("Count title similarity parameter...")
    temp1 = items[['itemID', 'title']].rename(columns={
         'itemID': 'itemID_1',
         'title': 'title_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'title']].rename(columns={
         'itemID': 'itemID_2',
         'title': 'title_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    vectorizer = prepareVectorizer_v2()
    table['title_proc_similarity_v2'] = table.apply(title_similarity, args=(vectorizer,), axis=1)
    table = table.drop(['title_1'], axis=1)
    table = table.drop(['title_2'], axis=1)
    print(table['title_proc_similarity_v2'].describe())
    return table


def count_proc_description_similar(table, items):
    print("Count description similarity parameter...")
    temp1 = items[['itemID', 'description']].rename(columns={
         'itemID': 'itemID_1',
         'description': 'description_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'description']].rename(columns={
         'itemID': 'itemID_2',
         'description': 'description_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    vectorizer = prepareVectorizer()
    table['description_proc_similarity'] = table.apply(description_similarity, args=(vectorizer,), axis=1)
    table = table.drop(['description_1'], axis=1)
    table = table.drop(['description_2'], axis=1)
    print(table['description_proc_similarity'].describe())
    return table


def count_proc_description_similar_v2(table, items):
    print("Count description similarity parameter...")
    temp1 = items[['itemID', 'description']].rename(columns={
         'itemID': 'itemID_1',
         'description': 'description_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'description']].rename(columns={
         'itemID': 'itemID_2',
         'description': 'description_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    vectorizer = prepareVectorizer_v2()
    table['description_proc_similarity_v2'] = table.apply(description_similarity, args=(vectorizer,), axis=1)
    table = table.drop(['description_1'], axis=1)
    table = table.drop(['description_2'], axis=1)
    print(table['description_proc_similarity_v2'].describe())
    return table


def count_proc_title_levinshtein(table, items):
    print("Count title levinshtein parameter...")
    temp1 = items[['itemID', 'title']].rename(columns={
        'itemID': 'itemID_1',
        'title': 'title_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'title']].rename(columns={
        'itemID': 'itemID_2',
        'title': 'title_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table.fillna('', inplace=True)
    table['title_proc_levinshtein'] = table.apply(title_levinshtein, axis=1)
    table = table.drop(['title_1'], axis=1)
    table = table.drop(['title_2'], axis=1)
    print(table['title_proc_levinshtein'].describe())
    return table


def count_proc_descr_levinshtein(table, items):
    print("Count descr levinshtein parameter...")
    temp1 = items[['itemID', 'description']].rename(columns={
        'itemID': 'itemID_1',
        'description': 'description_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'description']].rename(columns={
        'itemID': 'itemID_2',
        'description': 'description_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table.fillna('', inplace=True)
    table['descr_proc_levinshtein'] = table.apply(descr_levinshtein, axis=1)
    table = table.drop(['description_1'], axis=1)
    table = table.drop(['description_2'], axis=1)
    print(table['descr_proc_levinshtein'].describe())
    return table


def count_proc_title_damerau_levenshtein(table, items):
    print("Count title damerau levenshtein parameter...")
    temp1 = items[['itemID', 'title']].rename(columns={
        'itemID': 'itemID_1',
        'title': 'title_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'title']].rename(columns={
        'itemID': 'itemID_2',
        'title': 'title_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table.fillna('', inplace=True)
    table['title_proc_damerau_levenshtein_norm'] = table.apply(title_damerau_levenshtein_norm, axis=1)
    table['title_proc_damerau_levenshtein'] = table.apply(title_damerau_levenshtein, axis=1)
    table = table.drop(['title_1'], axis=1)
    table = table.drop(['title_2'], axis=1)
    print(table['title_proc_damerau_levenshtein_norm'].describe())
    print(table['title_proc_damerau_levenshtein'].describe())
    return table


def count_proc_descr_damerau_levenshtein(table, items):
    print("Count descr damerau levenshtein parameter...")
    temp1 = items[['itemID', 'description']].rename(columns={
        'itemID': 'itemID_1',
        'description': 'description_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'description']].rename(columns={
        'itemID': 'itemID_2',
        'description': 'description_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table.fillna('', inplace=True)
    table['descr_proc_damerau_levenshtein_norm'] = table.apply(descr_damerau_levenshtein_norm, axis=1)
    table['descr_proc_damerau_levenshtein'] = table.apply(descr_damerau_levenshtein, axis=1)
    table = table.drop(['description_1'], axis=1)
    table = table.drop(['description_2'], axis=1)
    print(table['descr_proc_damerau_levenshtein_norm'].describe())
    print(table['descr_proc_damerau_levenshtein'].describe())
    return table


def count_proc_title_jaro_winkler(table, items):
    print("Count title jaro winkler parameter...")
    temp1 = items[['itemID', 'title']].rename(columns={
        'itemID': 'itemID_1',
        'title': 'title_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'title']].rename(columns={
        'itemID': 'itemID_2',
        'title': 'title_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table.fillna('', inplace=True)
    table['title_jaro_winkler_t1'] = table.apply(title_jaro_winkler_t1, axis=1)
    table['title_jaro_winkler_t2'] = table.apply(title_jaro_winkler_t2, axis=1)
    table = table.drop(['title_1'], axis=1)
    table = table.drop(['title_2'], axis=1)
    print(table['title_jaro_winkler_t1'].describe())
    print(table['title_jaro_winkler_t2'].describe())
    return table


def count_proc_descr_jaro_winkler(table, items):
    print("Count descr jaro_winkler parameter...")
    temp1 = items[['itemID', 'description']].rename(columns={
        'itemID': 'itemID_1',
        'description': 'description_1',
    })
    table = pd.merge(table, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = items[['itemID', 'description']].rename(columns={
        'itemID': 'itemID_2',
        'description': 'description_2',
    })
    table = pd.merge(table, temp2, how='left', on='itemID_2', left_index=True)
    table.fillna('', inplace=True)
    table['descr_jaro_winkler_t1'] = table.apply(descr_jaro_winkler_t1, axis=1)
    table['descr_jaro_winkler_t2'] = table.apply(descr_jaro_winkler_t2, axis=1)
    table = table.drop(['description_1'], axis=1)
    table = table.drop(['description_2'], axis=1)
    print(table['descr_jaro_winkler_t1'].describe())
    print(table['descr_jaro_winkler_t2'].describe())
    return table


def prep_train_titles(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_title_similar(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/titles_info_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_levinshtein_title(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:100000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_title_levinshtein(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/titles_levinshtein_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_levinshtein_description(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_descr_levinshtein(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/descr_levinshtein_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_description(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_description_similar(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/description_info_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_titles(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_title_similar(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/titles_info_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_levinshtein_title(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_title_levinshtein(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/titles_levinshtein_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_levinshtein_descr(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_descr_levinshtein(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/descr_levinshtein_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_description(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_description_similar(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/description_info_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_damerau_levenshtein_title(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:100000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_title_damerau_levenshtein(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/titles_damerau_levenshtein_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_damerau_levenshtein_title(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_title_damerau_levenshtein(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/titles_damerau_levenshtein_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_damerau_levenshtein_description(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_descr_damerau_levenshtein(pairs, items)

    # print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/descr_damerau_levenshtein_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_damerau_levenshtein_descr(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_descr_damerau_levenshtein(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/descr_damerau_levenshtein_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_jaro_winkler_title(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:100000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_title_jaro_winkler(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/titles_jaro_winkler_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_jaro_winkler_title(testing = 1):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_title_jaro_winkler(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/titles_jaro_winkler_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_jaro_winkler_description(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_descr_jaro_winkler(pairs, items)

    # print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/descr_jaro_winkler_train.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_jaro_winkler_description(testing = 1):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_descr_jaro_winkler(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/descr_jaro_winkler_test.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


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
            out.write(line)
            check_array[(item_1, item_2)] = 1
        else:
            # print('Dublicate: ', item_1, item_2)
            count += 1

    print('Removed lines: {}'.format(count))
    out.close()
    f.close()


def prep_train_titles_v2(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_title_similar_v2(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/titles_info_train_v2.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_titles_v2(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_title_similar_v2(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/titles_info_test_v2.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_train_description_v2(testing = 1):

    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
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

    print("Load ItemPairs_train.csv")
    # ipairs = "../input/ItemPairs_train.csv"
    ipairs = "../modified_data/ItemPairs_train_with_additional_pairs.csv"
    if testing == 1:
        pairs = pd.read_csv(ipairs, dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv(ipairs, dtype=types1)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)

    print('Train length: {}'.format(len(pairs.index)))
    train = count_proc_description_similar_v2(pairs, items)

    print(train.describe())
    if testing == 0:
        print('Saving train...')
        train.to_csv("../modified_data/description_info_train_v2.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))


def prep_test_description_v2(testing = 0):

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

    print('Test length: {}'.format(len(pairs.index)))
    test = count_proc_description_similar_v2(pairs, items)

    print(test.describe())
    if testing == 0:
        print('Saving test...')
        test.to_csv("../modified_data/description_info_test_v2.csv", index=False)
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))


if 0:
    fix_csv("../modified_data/titles_info_train.csv", "../modified_data/titles_info_train_fixed.csv")
    fix_csv("../modified_data/description_info_train.csv", "../modified_data/description_info_train_fixed.csv")
    fix_csv("../modified_data/images_params_train.csv", "../modified_data/images_params_train_fixed.csv")
    fix_csv("../modified_data/images_params_test.csv", "../modified_data/images_params_test_fixed.csv")
    fix_csv("../modified_data/titles_levinshtein_train.csv", "../modified_data/titles_levinshtein_train_fixed.csv")
    fix_csv("../modified_data/descr_levinshtein_train.csv", "../modified_data/descr_levinshtein_train_fixed.csv")
    fix_csv("../modified_data/titles_damerau_levenshtein_train.csv", "../modified_data/titles_damerau_levenshtein_train_fixed.csv")
    fix_csv("../modified_data/descr_damerau_levenshtein_train.csv", "../modified_data/descr_damerau_levenshtein_train_fixed.csv")
    fix_csv("../modified_data/titles_jaro_winkler_train.csv", "../modified_data/titles_jaro_winkler_train_fixed.csv")
    fix_csv("../modified_data/descr_jaro_winkler_train.csv", "../modified_data/descr_jaro_winkler_train_fixed.csv")
    fix_csv("../modified_data/json_text_sim_params_train.csv", "../modified_data/json_text_sim_params_train_fixed.csv")
    fix_csv("../modified_data/json_detailed_params_train.csv", "../modified_data/json_detailed_params_train_fixed.csv")
    fix_csv("../modified_data/titles_info_train_v2.csv", "../modified_data/titles_info_train_v2_fixed.csv")
    fix_csv("../modified_data/description_info_train_v2.csv", "../modified_data/description_info_train_v2_fixed.csv")
    fix_csv("../modified_data/json_same_params_train.csv", "../modified_data/json_same_params_train_fixed.csv")
    exit()

if 0:
    prep_train_titles(0)
    # prep_test_titles(0)
    prep_train_description(0)
    # prep_test_description(0)

    prep_train_levinshtein_title(0)
    # prep_test_levinshtein_title(0)
    prep_train_levinshtein_description(0)
    # prep_test_levinshtein_descr(0)

    prep_train_damerau_levenshtein_title(0)
    # prep_test_damerau_levenshtein_title(0)
    prep_train_damerau_levenshtein_description(0)
    # prep_test_damerau_levenshtein_descr(0)

    prep_train_jaro_winkler_title(0)
    # prep_test_jaro_winkler_title(0)
    prep_train_jaro_winkler_description(0)
    # prep_test_jaro_winkler_description(0)

    prep_train_titles_v2(0)
    prep_test_titles_v2(0)
    prep_train_description_v2(0)
    prep_test_description_v2(0)