import numpy as np
import pandas as pd
import math
import functools

"""
Purpose: contains helpers for computing numeric features on columns 
that are needed for semantic labeling. These features come from
a paper on semantic labeling, whose experiment source code can be found here:
https://github.com/oeg-upm/ttla
"""

def get_sequential(col_lst, for_other=False):
    qs = np.quantile(col_lst, [0.25, 0.5, 0.75])
    trimean = (qs[0] + 2 * qs[1] + qs[2]) / 4
    
    tstd = 0.0
    for x in col_lst:
        tstd += (x - trimean) ** 2
    
    tstd = tstd / len(col_lst)
    tstd = math.sqrt(tstd)
    
    if for_other:
        return (trimean, tstd)
    else:
        return {'sequential' : (trimean, tstd)}

def get_other(col_lst):
    return {'other' : get_sequential(col_lst, for_other=True)}

def get_cat(col_lst):
    dct  ={}
    for c in col_lst:
        if c in dct:
            dct[c] += 1
        else:
            dct[c] = 0
    
    uniq = len(dct.keys())
    dist = sorted([dct[k] / len(col_lst) for k in dct])
    
    return {'categorical' : (uniq, dist)}

def get_counts(col_lst):
    csqt = list(map(math.sqrt, col_lst))
    sqrts = get_sequential(csqt, for_other=True)
    return {'counts' : sqrts}

def get_fts_by_tp(col_lst : list, tp : str):
    if tp == 'sequential':
        return get_sequential(col_lst)
    
    if tp == 'categorical':
        return get_cat(col_lst)
        
    if tp == 'count':
        return get_counts(col_lst)
    
    if tp == 'other':
        return get_other(col_lst)
    
    raise Exception("Features not implemented for Type: {}".format(tp))

def check_seq(col_lst):
    colset = set(col_lst)
    mn = min(col_lst)
    mx = max(col_lst)
    fullset = set(range(mn, mx + 1))
    sqrt = math.sqrt(len(fullset))
    if len(fullset.intersection(colset)) >= sqrt:
        return True
    return False

def check_cat(col_lst):
    clen = len(col_lst)
    slen = len(set(col_lst))
    
    if clen <= math.sqrt(clen) and slen > 1:
        return True
    return False

def check_cnt(col_lst):
    perc95 = np.quantile(col_lst, 0.95)
    quarts = np.quantile(col_lst, [0.25, 0.5, 0.75])
    
    gt_cond = ((perc95 - quarts[1]) / quarts[1]) >= 2
    lh_cond = (1.5 * (quarts[2] - quarts[0]) + quarts[2] <= perc95)
    
    return lh_cond and gt_cond

def is_float(col_lst):
    are_ints = [int(x) == x for x in col_lst]
    contains_ints = functools.reduce(lambda a, b : a and b, are_ints)
    return (not contains_ints)
    

def check_type(col_lst):
    if not is_float(col_lst):
        int_cols = [int(c) for c in col_lst]
        if check_seq(int_cols):
            return 'sequential'
        elif check_cat(int_cols):
            return 'categorical'
        elif check_cnt(int_cols):
            return 'count'
        else:
            return 'other'
    
    return 'other'