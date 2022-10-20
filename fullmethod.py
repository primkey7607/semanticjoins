import os
import pickle
import pandas as pd
from semhelper import check_type, get_fts_by_tp
from ast import literal_eval
import numpy as np

"""
Purpose: Given an input dataset and a data lake, run the full pipeline.
"""

def check_models(tp, models, hierarchy):
    print("tp : {}".format(tp))
    print("models: {}".format(models))
    if tp in models:
        return tp
    
    tp_ind = hierarchy.index(tp)
    for new_tp in hierarchy[tp_ind:]:
        if new_tp in models:
            return new_tp
    
    #but it's also possible to have the last element in the hierarchy,
    #in which case, we should use the one immediately before.
    return hierarchy[tp_ind - 1]
    

def sem_classify(fname, is_test=False):
    #we organize the models by a hierarchy, so that if numeric types don't match
    #any type of FCM models we've trained,
    #we can still try the model whose numeric type might be closest to what we found
    hierarchy = ['other', 'sequential', 'counts', 'categorical']
    #first, load models into memory...
    cp_labels = {}
    fcm_models = {}
    centroid_names = {}
    for f in os.listdir():
        if f.startswith('fcm_') and f.endswith('model.pkl'):
            mtp = f[4:-9]
            with open(f, 'rb') as fh:
                model = pickle.load(fh)
            
            fcm_models[mtp] = model
            with open('fcm_' + mtp + 'centroids.txt', 'r') as fh:
                st = fh.read()
                centroid_names[mtp] = literal_eval(st)
    
    with open('max_numfts.txt', 'r') as fh:
        st = fh.read()
        mx_cats = literal_eval(st)
    
    
    #then, load the table as a list of columns
    if is_test:
        df = pd.read_csv(fname)
        cols = [c for c in df.columns if df.dtypes[c] != 'object' and 'Unnamed' not in c]
        df = df[cols]
    else:
        df = pd.read_csv(fname)
    
    #get rid of null-valued rows
    df = df[~df.isnull().any(axis=1)]
    
    for c in df.columns:
        col_lst = df[c].to_list()
        coltp = check_type(col_lst)
        numfts = get_fts_by_tp(col_lst, coltp)
        if coltp == 'categorical':
            #then, we need to pad with zeros
            ft_lst = list(numfts[coltp])
            if len(ft_lst[1]) < mx_cats:
                new_hist = ft_lst[1] + [0] * (mx_cats - len(ft_lst[1]))
                new_fts = [ft_lst[0]] + new_hist
            else:
                new_fts = ft_lst
        else:
            new_fts = list(numfts[coltp])
        ft_arr = np.array([new_fts])
        print(ft_arr)
        htp = check_models(coltp, fcm_models, hierarchy)
        raw_pred = fcm_models[htp].predict(ft_arr)
        #next, we need to translate the predictions from numbers to names
        pwnames = zip(raw_pred[0], centroid_names[htp])
        pwnames = sorted(pwnames, key=lambda x: x[0], reverse=True)
        #this line takes the top 3 columns. we can parametrize for top k.
        cp_labels[c] = pwnames[:3]
    
    return cp_labels

if __name__ == "__main__":
    #test classification using models on properties we already have
    print(sem_classify('busridertbl.csv', is_test=True))
    
    

