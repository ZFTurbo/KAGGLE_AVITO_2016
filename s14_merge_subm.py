# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import os
import zipfile
import numpy as np
from collections import defaultdict
np.random.seed(2016)


def merge_submissions(subm_list, out_file):
    res = defaultdict(float)
    first_line = ''
    for s in subm_list:
        f = open(s)
        first_line = f.readline()
        while 1:
            line = f.readline().strip()
            if line == '':
                break
            arr = line.split(',')
            res[arr[0]] += float(arr[1])

        f.close()

    num = len(subm_list)
    out = open(out_file, "w")
    out.write(first_line)
    for el in sorted(res):
        out.write(str(el) + ',')
        out.write(str(res[el]/num) + '\n')

    out.close()

    print('Creating zip-file...')
    z = zipfile.ZipFile(out_file + ".zip", "w", zipfile.ZIP_DEFLATED)
    z.write(out_file)
    z.close()


def merge_submissions_improve_first(subm_list, out_file):
    best = defaultdict(float)
    other = dict()
    res = dict()

    best_path = subm_list[0]
    other_path = subm_list[1:]

    # Read best submission
    f = open(best_path)
    first_line = f.readline()
    while 1:
        line = f.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        best[arr[0]] = float(arr[1])
    f.close()

    # Read all other submissions
    num = len(other_path)
    for i in range(num):
        other[i] = defaultdict(float)
        s = other_path[i]
        f = open(s)
        first_line = f.readline()
        while 1:
            line = f.readline().strip()
            if line == '':
                break
            arr = line.split(',')
            other[i][arr[0]] = float(arr[1])
        f.close()

    # Improve
    total = 0
    changed = 0
    for el in best:
        count_greater = 0
        count_lower = 0
        res[el] = 0.0
        min_val = 2.0
        max_val = -2.0
        for i in range(num):
            res[el] += other[i][el]
            if best[el] > 0.9:
                if other[i][el] > best[el]:
                    count_greater += 1
                    if other[i][el] < min_val:
                        min_val = other[i][el]
            if best[el] < 0.1:
                if other[i][el] < best[el]:
                    count_lower += 1
                    if other[i][el] > max_val:
                        max_val = other[i][el]
        if count_greater == num:
            res[el] = min_val
            changed += 1
        elif count_lower == num:
            res[el] = max_val
            changed += 1
        else:
            res[el] = best[el]
        total += 1

    print('Changed {} of {}'.format(changed, total))
    out = open(out_file, "w")
    out.write(first_line)
    for el in range(1044196):
        out.write(str(el) + ',')
        out.write(str(res[str(el)]) + '\n')

    out.close()

    print('Creating zip-file...')
    z = zipfile.ZipFile(out_file + ".zip", "w", zipfile.ZIP_DEFLATED)
    z.write(out_file)
    z.close()


if __name__ == '__main__':
    subm = []
    # LB: 0.93294
    # subm.append(os.path.join("subm", "submission_0.969687835643_2016-06-08-08-37.csv"))
    # LB: 0.93287
    # subm.append(os.path.join("subm", "submission_0.96862470875_2016-06-06-06-13.csv"))
    # subm.append(os.path.join("subm", "subm-0.93402.csv"))
    # subm.append(os.path.join("subm", "subm-0.93175.csv"))
    # subm.append(os.path.join("subm", "subm-0.94272.csv"))
    # subm.append(os.path.join("subm", "subm-0.94292.csv"))
    # subm.append(os.path.join("subm", "subm-0.94499.csv"))
    # subm.append(os.path.join("subm", "subm-0.94162.csv"))
    if 0:
        subm.append(os.path.join("..", "run_0.94272", "submission_0.976437314005_2016-07-05-20-25.csv"))
        subm.append(os.path.join("..", "run_0.94292", "submission_0.97712283726_2016-07-05-22-33.csv"))
        subm.append(os.path.join("..", "run_0.94453", "submission_0.976687365593_2016-07-02-03-28.csv"))
        subm.append(os.path.join("..", "run_0.94xv4", "submission_0.99324157307_2016-07-10-18-06.csv"))
        subm.append(os.path.join("..", "run_0.94xv5", "submission_0.996107957693_2016-07-10-20-26.csv"))

    if 0:
        subm.append(os.path.join("subm", "subm-0.94499.csv"))
        subm.append(os.path.join("subm", "subm-0.94349.csv"))

    subm.append(os.path.join("subm", "subm-0.94700.csv"))
    subm.append(os.path.join("subm", "subm-0.94675.csv"))

    out_path = os.path.join("subm", "submission_merge.csv")
    merge_submissions(subm, out_path)
    # merge_submissions_improve_first(subm, out_path)

# Merge of 5 best: 0.93000 - bad (was 0.93051)
# Merge only when all is better or worse: 0.92987 - bad (was 0.93051)
# Use min or max if all greater or lower: 0.92927 - very bad (was 0.93051)
# Use min, max for edge cases: 0.93061 - improved from 0.93051
# 0.93294 + 0.93287 = 0.93339
# 0.93402 + 0.93175 = 0.93693
# 0.94453 + 0.94132 = 0.94700
# 0.94453 + 0.94272 + 0.94292 = 0.94489
# 0.94499 + 0.94162 = 0.94703
# 5 Best = 0.94470
# 0.94499 + 0.94349 = 0.94633 (Models with Id features)
# 0.94633 + 0.94132 = 0.94615
# 0.94700 + 0.94675 = 0.94732