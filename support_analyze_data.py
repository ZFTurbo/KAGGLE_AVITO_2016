# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import defaultdict
import time
import copy
import json
import nltk, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('all')
# exit()


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_numpy(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

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


def find_same_with_diff_title(testing = 1):

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
    if testing == 1:
        pairs = pd.read_csv("../input/ItemPairs_train.csv", dtype=types1)[0:10000]
    else:
        pairs = pd.read_csv("../input/ItemPairs_train.csv", dtype=types1)
    # Add 'id' column for easy merge
    pairs['id'] = pairs.index.astype(int)
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)
    items.fillna(-1, inplace=True)

    train = pairs
    train = train.drop(['generationMethod'], axis=1)

    print('Merge item 1...')
    item1 = items[['itemID', 'title']]

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'title': 'title_1',
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[['itemID', 'title']]

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'title': 'title_2',
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)
    train = train[train['title_1'] != train['title_2']]

    print('Saving train...')
    train.to_csv("../modified_data/analysis_train_same_diff_title.csv", index=False)
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))

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


def prepareVectorizer():
    vectorizer = TfidfVectorizer(tokenizer=normalize, lowercase=True)
    return vectorizer


def cosine_sim(text1, text2, vectorizer):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except:
        return 0.0
    return ((tfidf * tfidf.T).A)[0, 1]


def test_vectorizer():
    vect = prepareVectorizer()
    text1 = 'Super Nintendo Batman Returns'
    text2 = 'Super Nintendo Donkey Kong Country'
    print(normalize(text1))
    print(normalize(text2))
    print(cosine_sim(text1, text2, vect))
    text1 = 'Yamaha r6'
    text2 = 'Yamaha R6'
    print(normalize(text1))
    print(normalize(text2))
    print(cosine_sim(text1, text2, vect))
    text1 = 'Лыжные ботинки'
    text2 = 'Ботинки для лыж'
    print(normalize(text1))
    print(normalize(text2))
    print(cosine_sim(text1, text2, vect))
    text1 = 'Полупальто'
    text2 = 'Полупальто sela'
    print(normalize(text1))
    print(normalize(text2))
    print(cosine_sim(text1, text2, vect))
    exit()

exit()
test_vectorizer()
exit()
print('Yamaha r6'.lower(), 'Yamaha R6'.lower())
print('Парник подснежник в полном сборе 4м'.lower())
exit()
find_same_with_diff_title(1)

