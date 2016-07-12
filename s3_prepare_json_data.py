# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
import pickle
from transliterate import translit
import time
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import nltk, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


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


def get_unique_json():
    jsonDict = dict()
    jsonDictTopLevel = dict()
    print('Reading data...')
    # items = pd.read_csv("../input/ItemInfo_train.csv")[0:10000]
    items = pd.read_csv("../input/ItemInfo_train.csv")
    items.fillna(-1, inplace=True)
    jsonOnly = items[['attrsJSON']]
    print('Proccessing data...')
    for i, row in jsonOnly.iterrows():
        # print(str(row['attrsJSON']))
        if row['attrsJSON'] == -1:
            continue
        data = json.loads(str(row['attrsJSON']))
        # print(data)
        for el in data:
            if el in jsonDict:
                if data[el] in jsonDict[el]:
                    jsonDict[el][data[el]] += 1
                else:
                    jsonDict[el][data[el]] = 1
            else:
                jsonDict[el] = dict()
                jsonDict[el][data[el]] = 1

            if el in jsonDictTopLevel:
                jsonDictTopLevel[el] += 1
            else:
                jsonDictTopLevel[el] = 1

    skip_cats = ['Опыт работы', 'Образование']

    print('Writing data...')
    out = open("../modified_data/json_analysis.txt", "w", encoding="utf-8")
    for el in sorted([(value, key) for (key, value) in jsonDictTopLevel.items()], reverse=True):
        value = el[0]
        key = el[1]
        if key in skip_cats:
            continue
        if value > 100:
            out.write('Cat: {} Num: {} Unique: {}\n'.format(key, value, len(jsonDict[key])))
            out.write(str(sorted(jsonDict[key])) + '\n')
            # print('Cat: {} Num: {} Unique: {}'.format(el, jsonDictTopLevel[el], len(jsonDict[el])))
            # print(dict(jsonDict[el]))
    out.close()
    return jsonDict, jsonDictTopLevel


def get_filled_table(input_path, type='train'):
    if type == 'train':
        items = pd.read_csv("../input/ItemInfo_train.csv")
    else:
        items = pd.read_csv("../input/ItemInfo_test.csv")
    items.fillna(-1, inplace=True)
    jsonOnly = items[['itemID', 'attrsJSON']]

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }

    print("Load pairs.csv")
    # pairs = pd.read_csv(input_path, dtype=types1)[0:10000]
    pairs = pd.read_csv(input_path, dtype=types1)

    temp1 = jsonOnly[['itemID', 'attrsJSON']].rename(columns={
         'itemID': 'itemID_1',
         'attrsJSON': 'attrsJSON_1',
    })
    pairs = pd.merge(pairs, temp1, how='left', on='itemID_1', left_index=True)
    temp2 = jsonOnly[['itemID', 'attrsJSON']].rename(columns={
         'itemID': 'itemID_2',
         'attrsJSON': 'attrsJSON_2',
    })
    pairs = pd.merge(pairs, temp2, how='left', on='itemID_2', left_index=True)
    return pairs


def get_param_name(key):
    str = translit(key, 'ru', reversed=True)
    str = 'jsn_' + str.lower().replace(' ', '_').strip()
    str = str.replace('\'', '')
    str = str.replace('/', '_')
    str = str.replace('-', '_')
    str = str.replace('(', '_')
    str = str.replace(')', '_')
    return str


def getDataInRightFormat(key, value, jsonSorted):
    # Special case 1
    if key == 'Год выпуска':
        if len(value) != 4:
            return 1
        else:
            return int(value)

    # Special case 2
    if key in ['Объём двигателя']:
        if len(value) != 3:
            return 7.0
        else:
            return float(value)

    # Сделать Пробег
    # ....

    # Int cases
    if key in ['Мощность двигателя', 'Этажей в доме', 'Количество дверей', 'Возраст', 'Этаж', 'Диаметр',
               'Ширина профиля', 'Высота профиля', 'Количество отверстий', 'Расстояние до города', 'Диски',
               'Кол-во спальных мест', 'Размер комиссии', 'Название новостройки', 'Корпус', '']:
        if key == 'Диски':
            value = value.replace('"', '')
        if value == '16+':
            return 16
        if value == '5+':
            return 5.5
        return int(value)

    # Float cases
    if key in ['Площадь', 'Диаметр расположения отверстий', 'Площадь участка', 'Площадь дома', 'Ширина обода',
               'Вылет (ET)', 'Площадь комнаты', '']:
        if value == '> 30':
            return 30.1
        return float(value)

    return jsonSorted.index(value)


def print_local_values(out, key, data1, data2, jsonSorted):
    if key in data1:
        if data1[key] in jsonSorted:
            st = getDataInRightFormat(key, data1[key], jsonSorted)
            out.write(',' + str(st))
        else:
            # print('Check: ' + str(data1[key]))
            out.write(',-1')
    else:
        out.write(',-1')
    if key in data2:
        if data2[key] in jsonSorted:
            st = getDataInRightFormat(key, data2[key], jsonSorted)
            out.write(',' + str(st))
        else:
            # print('Check: ' + str(data1[key]))
            out.write(',-1')
    else:
        out.write(',-1')


def print_local_values_same(out, key, data1, data2, jsonSorted):
    if key not in data1 and key not in data2:
        out.write(',-1')
    elif key not in data1 or key not in data2:
        out.write(',0')
    else:
        st1 = '-1'
        if data1[key] in jsonSorted:
            st1 = getDataInRightFormat(key, data1[key], jsonSorted)
        st2 = '-2'
        if data2[key] in jsonSorted:
            st2 = getDataInRightFormat(key, data2[key], jsonSorted)
        if st1 == st2:
            out.write(',1')
        else:
            out.write(',0')


def create_json_csv(jsonDict, jsonDictTopLevel, type='train'):
    trim_const = 1
    if type == 'train':
        out_path = "../modified_data/json_detailed_params_train.csv"
        input_path = "../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv"
    else:
        out_path = "../modified_data/json_detailed_params_test.csv"
        input_path = "../input/ItemPairs_test.csv"
    print('Get table...')
    table = get_filled_table(input_path, type)
    print('Sort dict...')
    jsonSorted = dict()
    for el in jsonDict:
        jsonSorted[el] = sorted(jsonDict[el])

    # Remove unneeded
    skip_cats = ['Опыт работы', 'Образование', 'Адрес', 'Забронированные даты', 'Модель',
                 'Знание языков', 'Отчёт Автокод', 'Кадастровый номер', 'Номер свидетельства ТС',
                 'VIN-номер', 'Корпус', 'Ссылка на документацию', 'Корпус / очередь', 'Страна', 'Название новостройки',
                 'Кадастровый номер участка', 'Адрес компании']
    jsonSimplified = dict()
    for el in jsonDictTopLevel:
        if el not in skip_cats:
            if jsonDictTopLevel[el] > trim_const:
                jsonSimplified[el] = jsonDictTopLevel[el]

    sortedIter = sorted([(value, key) for (key, value) in jsonSimplified.items()], reverse=True)
    print(sortedIter)

    print('Write table in CSV ...')
    out = open(out_path, "w", encoding='UTF-8')
    out.write('itemID_1,itemID_2')
    # print header
    for el in sortedIter:
        key = el[1]
        nm = get_param_name(key)
        out.write(',' + nm + '_1')
        out.write(',' + nm + '_2')
        # print('\'' + nm + '\': np.dtype(np.float32),')
    out.write('\n')

    for i, row in table.iterrows():
        out.write(str(row['itemID_1']))
        out.write(',')
        out.write(str(row['itemID_2']))
        if row['attrsJSON_1'] == -1:
            data1 = dict()
        else:
            data1 = json.loads(str(row['attrsJSON_1']))
        if row['attrsJSON_2'] == -1:
            data2 = dict()
        else:
            data2 = json.loads(str(row['attrsJSON_2']))

        for el in sortedIter:
            key = el[1]
            print_local_values(out, key, data1, data2, jsonSorted[key])

        out.write('\n')
    out.close()


def check_most_absent_in_test(jsonDict, jsonDictTopLevel):
    trim_const = 1
    input_path = "../input/ItemPairs_test.csv"
    print('Get table...')
    table = get_filled_table(input_path, 'test')

    print('Sort dict...')
    jsonSorted = dict()
    for el in jsonDict:
        jsonSorted[el] = sorted(jsonDict[el])
    sortedIter = sorted([(value, key) for (key, value) in dict(jsonDictTopLevel).items()], reverse=True)

    absent = defaultdict(int)
    for i, row in table.iterrows():
        if row['attrsJSON_1'] == -1:
            data1 = dict()
        else:
            data1 = json.loads(str(row['attrsJSON_1']))
        if row['attrsJSON_2'] == -1:
            data2 = dict()
        else:
            data2 = json.loads(str(row['attrsJSON_2']))

        for el in sortedIter:
            value = el[0]
            key = el[1]
            if value > trim_const:
                if key in data1:
                    if data1[key] not in jsonSorted[key]:
                        # print(key, data1[key])
                        absent[key] += 1

    print(dict(absent))
    # {'Забронированные даты': 94, 'Образование': 7914, 'Модель': 88, 'Знание языков': 174, 'Площадь': 638,
    # 'Марка': 3, 'Отчёт Автокод': 318, 'Кадастровый номер': 577, 'Расстояние до города': 2, 'Возраст': 5,
    # 'Номер свидетельства ТС': 1032, 'VIN-номер': 1876, 'Площадь участка': 27, 'Корпус': 316,
    # 'Мощность двигателя': 16, 'Размер комиссии': 5, 'Ширина профиля': 4, 'Этажей в доме': 11,
    # 'Гражданство': 13, 'Порода': 6, 'Ссылка на документацию': 85, 'Площадь дома': 161, 'Этаж': 9,
    # 'Корпус / очередь': 23, 'Страна': 96, 'Название новостройки': 713, 'Опыт работы': 7057, 'Адрес': 34560,
    # 'Кадастровый номер участка': 743, 'Адрес компании': 1485, 'Вылет (ET)': 7, 'Площадь комнаты': 57}


def create_json_text_similarity(type='train'):
    if type == 'train':
        input_path = "../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv"
        # input_path = "../input/ItemPairs_train.csv"
        # out_path = "../modified_data/json_text_sim_params_train.csv"
        out_path = "../orig_features/train_json_sim_param.csv"
    else:
        input_path = "../input/ItemPairs_test.csv"
        # out_path = "../modified_data/json_text_sim_params_test.csv"
        out_path = "../orig_features/test_json_sim_param.csv"

    print('Get table...')
    table = get_filled_table(input_path, type)
    only_cats = ['Опыт работы', 'Образование', 'Адрес', 'Забронированные даты', 'Модель',
                 'Знание языков', 'Отчёт Автокод', 'Кадастровый номер', 'Номер свидетельства ТС',
                 'VIN-номер', 'Корпус', 'Ссылка на документацию', 'Корпус / очередь', 'Страна', 'Название новостройки',
                 'Кадастровый номер участка', 'Адрес компании']

    vectorizer = prepareVectorizer()
    print('Write table in CSV ...')
    out = open(out_path, "w", encoding='UTF-8')
    out.write('itemID_1,itemID_2')
    # print header
    for key in only_cats:
        nm = get_param_name(key)
        out.write(',' + nm + '_dam_lev_norm')
    out.write(',address_tdidf')
    out.write('\n')

    for i, row in table.iterrows():
        out.write(str(row['itemID_1']))
        out.write(',')
        out.write(str(row['itemID_2']))
        if row['attrsJSON_1'] == -1:
            data1 = dict()
        else:
            data1 = json.loads(str(row['attrsJSON_1']))
        if row['attrsJSON_2'] == -1:
            data2 = dict()
        else:
            data2 = json.loads(str(row['attrsJSON_2']))

        for key in only_cats:
            if key not in data1 and key not in data2:
                out.write(',-1')
            else:
                str1 = ''
                str2 = ''
                if key in data1:
                    str1 = data1[key]
                if key in data2:
                    str2 = data2[key]
                val = normalized_damerau_levenshtein_distance(str1, str2)
                out.write(',' + str(val))

        # Адрес считаем tfidf
        for key in ['Адрес']:
            if key not in data1 and key not in data2:
                out.write(',-1')
            else:
                str1 = ''
                str2 = ''
                if key in data1:
                    str1 = data1[key]
                if key in data2:
                    str2 = data2[key]
                val = cosine_sim(str1, str2, vectorizer)
                out.write(',' + str(val))

        out.write('\n')
    out.close()


def create_json_csv_same(jsonDict, jsonDictTopLevel, type='train'):
    trim_const = 1
    if type == 'train':
        input_path = "../modified_data/ItemPairs_train_with_additional_pairs_fixed.csv"
        # input_path = "../input/ItemPairs_train.csv"
        # out_path = "../modified_data/json_same_params_train.csv"
        out_path = "../orig_features/train_json_same_param.csv"
    else:
        input_path = "../input/ItemPairs_test.csv"
        # out_path = "../modified_data/json_same_params_test.csv"
        out_path = "../orig_features/test_json_same_param.csv"

    print('Get table...')
    table = get_filled_table(input_path, type)
    print('Sort dict...')
    jsonSorted = dict()
    for el in jsonDict:
        jsonSorted[el] = sorted(jsonDict[el])

    # Remove unneeded
    skip_cats = ['Опыт работы', 'Образование', 'Адрес', 'Забронированные даты', 'Модель',
                 'Знание языков', 'Отчёт Автокод', 'Кадастровый номер', 'Номер свидетельства ТС',
                 'VIN-номер', 'Корпус', 'Ссылка на документацию', 'Корпус / очередь', 'Страна', 'Название новостройки',
                 'Кадастровый номер участка', 'Адрес компании']
    jsonSimplified = dict()
    for el in jsonDictTopLevel:
        if el not in skip_cats:
            if jsonDictTopLevel[el] > trim_const:
                jsonSimplified[el] = jsonDictTopLevel[el]

    sortedIter = sorted([(value, key) for (key, value) in jsonSimplified.items()], reverse=True)
    print(sortedIter)

    print('Write table in CSV ...')
    out = open(out_path, "w", encoding='UTF-8')
    out.write('itemID_1,itemID_2')
    # print header
    for el in sortedIter:
        key = el[1]
        nm = get_param_name(key)
        out.write(',' + nm + '_same')
        # print('\'' + nm + '\': np.dtype(np.float32),')
    out.write('\n')

    for i, row in table.iterrows():
        out.write(str(row['itemID_1']))
        out.write(',')
        out.write(str(row['itemID_2']))
        if row['attrsJSON_1'] == -1:
            data1 = dict()
        else:
            data1 = json.loads(str(row['attrsJSON_1']))
        if row['attrsJSON_2'] == -1:
            data2 = dict()
        else:
            data2 = json.loads(str(row['attrsJSON_2']))

        for el in sortedIter:
            key = el[1]
            print_local_values_same(out, key, data1, data2, jsonSorted[key])

        out.write('\n')
    out.close()


start_time = time.time()
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
cache_path = 'cache/json.cache.pickle.dat'
if not os.path.isfile(cache_path):
    jsonDict, jsonDictTopLevel = get_unique_json()
    cache_data((jsonDict, jsonDictTopLevel), cache_path)
else:
    (jsonDict, jsonDictTopLevel) = restore_data(cache_path)

create_json_csv(jsonDict, jsonDictTopLevel, 'train')
create_json_csv(jsonDict, jsonDictTopLevel, 'test')
create_json_csv_same(jsonDict, jsonDictTopLevel, 'train')
create_json_csv_same(jsonDict, jsonDictTopLevel, 'test')
# check_most_absent_in_test(jsonDict, jsonDictTopLevel)
create_json_text_similarity('train')
create_json_text_similarity('test')

print('Run time: {} seconds'.format(round(time.time() - start_time, 2)))

'''
[(1448620, 'Вид товара'), (556781, 'Вид одежды'), (530765, 'Предмет одежды'), (491484, 'Размер'), (431023, 'Тип товара'),
(264839, 'Тип кузова'), (264839, 'Тип двигателя'), (264839, 'Тип автомобиля'), (264839, 'Коробка передач'),
(264838, 'Марка'), (264828, 'Привод'), (264695, 'Год выпуска'), (264673, 'Объём двигателя'), (264487, 'Модель'),
(255255, 'Цвет'), (255248, 'Руль'), (253259, 'Вид телефона'), (249167, 'Тип объявления'), (245519, 'Мощность двигателя'),
(227191, 'Адрес'), (224126, 'Пробег'), (188349, 'Этажей в доме'), (180108, 'Состояние'), (179819, 'Количество дверей'),
(175485, 'Площадь'), (175158, 'Сфера деятельности'), (175158, 'График работы'), (175108, 'Опыт работы'), (169586, 'Пол'),
(169586, 'Образование'), (168662, 'Переезд'), (168662, 'Готовность к командировкам'), (168211, 'Гражданство'),
(159267, 'Возраст'), (156613, 'Этаж'), (156577, 'Тип дома'), (146591, 'Диаметр'), (146196, 'Вид объекта'),
(140271, 'Количество комнат'), (132680, 'Вид запчасти'), (106844, 'Сезонность'), (99949, 'Вид оборудования'),
(96803, 'Вид услуги'), (95617, 'Ширина профиля'), (94995, 'Высота профиля'), (78890, 'Срок аренды'),
(78191, 'Тип диска'), (68247, 'Порода'), (64433, 'Количество отверстий'), (59703, 'Тип услуги'),
(53592, 'Залог'), (53591, 'Комиссия'), (51283, 'Расстояние до города'), (50675, 'Диаметр расположения отверстий'),
(48111, 'Усилитель руля'), (46288, 'Вид техники'), (44706, 'зеркал'), (42351, 'Салон'), (42290, 'центральный замок'),
(41893, 'Электростеклоподъемники'), (40463, 'фронтальные'), (40347, 'Управление климатом'),
(38157, 'Владельцев по ПТС'), (37951, 'MP3'), (37233, 'передних сидений'), (35729, 'бортовой компьютер'),
(33769, 'Плита'), (33666, 'сигнализация'), (33272, 'Холодильник'), (31745, 'Площадь участка'), (31745, 'Площадь дома'),
(31730, 'Материал стен'), (30815, 'Ширина обода'), (30433, 'CD/DVD/Blu-ray'), (29882, 'радио'), (29086, 'Диски'),
(28433, 'Аудиосистема'), (28359, 'заднего стекла'), (28011, 'Телевизор'), (27340, 'Кол-во спальных мест'),
(27333, 'Кол-во кроватей'), (27144, 'Фары'), (26242, 'Стиральная машина'), (24984, 'Вылет (ET)'), (23789, 'USB'),
(22542, 'противотуманные'), (22434, 'иммобилайзер'), (21595, 'Категория земель'), (21502, 'VIN-номер'),
(21472, 'Балкон / лоджия'), (21130, 'Вид велосипеда'), (20619, 'антиблокировочная система тормозов (ABS)'),
(19997, 'Кабельное / цифровое ТВ'), (19384, 'Можно с детьми'), (19315, 'боковые передние'), (18888, 'AUX'),
(18200, 'Площадь комнаты'), (18200, 'Комнат в квартире'), (18023, 'зимние шины в комплекте'), (17834, 'датчик света'),
(17211, 'Микроволновка'), (16954, 'Wi-Fi'), (16704, 'датчик дождя'), (15941, 'Размер комиссии'),
(15728, 'круиз-контроль'), (15312, 'Знание языков'), (14161, 'есть сервисная книжка'), (12059, 'Утюг'),
(11667, 'парктроник задний'), (11045, 'кожаный руль'), (10841, 'антипробуксовочная система (ASR)'),
(10707, 'управление на руле'), (10216, 'Парковочное место'), (9874, 'омыватели фар'), (9727, 'Вид животного'),
(9426, 'обслуживался у дилера'), (9418, 'Размерность цены'), (9326, 'система курсовой устойчивости (ESP/ESC/DSC)'),
(8318, 'Охрана'), (8227, 'Отчёт Автокод'), (8155, 'Bluetooth'), (7906, 'Тип гаража'),
(7651, 'система распределения тормозных усилий (EBD/EBV)'), (7532, 'Местонахождение'), (7517, 'шторки'),
(7212, 'Номер свидетельства ТС'), (6897, 'атермальное остекление'), (6664, 'боковые задние'),
(6509, 'Можно с питомцами'), (5868, 'GPS-навигатор'), (5620, 'Фен'), (5596, 'складывания зеркал'), (5278, 'на гарантии'),
(5224, 'Кондиционер'), (5193, 'коленные'), (5066, 'задних сидений'), (5013, 'Вид мотоцикла'), (4775, 'парктроник передний'),
(4687, 'люк'), (4434, 'Разрешение на работу в России'), (4199, 'система экстренного торможения (EBA/BAS/BA)'),
(3935, 'Можно курить'), (3715, 'Адрес компании'), (3643, 'видео'), (3518, 'TV'), (3440, 'камера заднего вида'),
(3141, 'рулевой колонки'), (2923, 'Показывать ссылку на полный отчёт Автокода'), (2849, 'сабвуфер'), (2818, 'Кадастровый номер участка'),
(2754, 'электронная блокировка дифференциала (EDS/XDS/ETS)'), (2705, 'руля'), (2572, 'Кадастровый номер'),
(2445, 'адаптивное освещение'), (2336, 'Можно для мероприятий'), (2236, 'Название новостройки'), (2060, 'Вид бизнеса'),
(2036, 'Вид устройства'), (1790, 'Страна'), (1453, 'Корпус'), (1157, 'Забронированные даты'), (1094, 'Баня / сауна'),
(957, 'система контроля слепых зон'), (876, 'Класс здания'), (826, 'спутник'), (781, 'Камин'), (679, 'автоматический парковщик'),
(412, 'Тип машиноместа'), (347, 'Ось'), (322, 'система обнаружения пешеходов (PDS)'), (286, 'Корпус / очередь'),
(273, 'Ссылка на документацию'), (220, 'Бассейн'), (95, 'Тип техники'), (32, 'Вес'), (17, 'Модель устройства'),
(6, 'Тип телевизора'), (6, 'Модель телевизора'), (5, 'Модель стиральной машины')]
'''
