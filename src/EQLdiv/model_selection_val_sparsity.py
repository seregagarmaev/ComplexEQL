import numpy as np
from operator import itemgetter


'''
expects a file with one row per network and columns reporting the parameters and sparsity and performance
First line should be the column names, #C col1 col2 col3...
then one additional comments line:  # extrapolation datasets etc
A sample file is in example_parameter_scan_result.txt

These are the typical columns is the file.
['k', 'iter', 'layers', 'epochs', 'nodes', 'lr', 'L1', 'L2', 'shortcut', 'batchsize', 'regstart', 'regend',
'id','dataset', 'gradient', 'numactive', 'bestnumactive', 'bestepoch','dups', 'inserts', 'runtime', 'extrapol1', 'extrapol2', 'extrapol3',
 'extrapolbest1', 'extrapolbest2', 'extrapolbest3', 'valerror', 'valerrorbest', 'testerror']
'''


def _normalize(arr):
    arr = np.asarray(arr, dtype=float)

    arr_min = np.min(arr)
    arr_max = np.max(arr)

    if arr_max == arr_min:
        return np.zeros_like(arr)

    return (arr - arr_min) / (arr_max - arr_min)


def select_instance(filename):
    value_dict = {}

    with open(filename, 'r') as f:
        k = 0
        lines = f.readlines()
        keys = lines[0].split()[1:]
        extrapolL = [x for x in keys if ("extrapol" in x and "best" not in x)]

        for key in keys:
            nums = []
            for line in lines[2:]:
                nums.append(line.split()[k])
            k += 1
            value_dict[key] = nums

    n_lines = 0

    for i in range(len(value_dict["id"])):
        value_dict["id"][i] = int(value_dict["id"][i])
        value_dict["nodes"][i] = int(value_dict["nodes"][i])
        value_dict["numactive"][i] = float(value_dict["numactive"][i])
        value_dict["iter"][i] = int(value_dict["iter"][i])
        value_dict["valerror"][i] = float(value_dict["valerror"][i])

        for key in extrapolL:
            value_dict[key][i] = float(value_dict[key][i])

        n_lines += 1

    print("lines: ", n_lines)

    active_ = []
    validation_ = []
    id_ = []
    extrapol_ = []

    has_extrapol2 = "extrapol2" in value_dict

    for i in range(n_lines):
        validation_.append(value_dict["valerror"][i])
        active_.append(value_dict["numactive"][i])
        if has_extrapol2:
            extrapol_.append(value_dict["extrapol2"][i])
        id_.append(value_dict["id"][i])

    active = _normalize(active_)
    validation = _normalize(validation_)

    norm_score = np.sqrt(active ** 2 + validation ** 2)

    if len(extrapol_) > 0:
        best_extrapol = sorted(zip(id_, extrapol_), key=itemgetter(1))[0]
        print((" best extrapolating model: (only for information):", best_extrapol))
    else:
        best_extrapol = None

    if len(extrapol_) > 0:
        score = list(zip(list(norm_score), id_, active_, validation_, extrapol_))
        score.sort(key=itemgetter(0))
        best_instance = score[0]
        print(("selected instance model: score: {} id: {} #active: {}\t val-error: {}\t extra-pol2-error: {}".format(*best_instance)))
        return dict(zip(['score', 'id', 'num_active', 'valerror', 'extrapol2'], best_instance))
    else:
        score = list(zip(list(norm_score), id_, active_, validation_))
        score.sort(key=itemgetter(0))
        best_instance = score[0]
        print(("selected instance model: score: {} id: {} #active: {}\t val-error: {}".format(*best_instance)))
        return dict(zip(['score', 'id', 'num_active', 'valerror'], best_instance))