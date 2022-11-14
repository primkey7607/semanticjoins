import os
import pickle
import pandas as pd
from semhelper import check_type, get_fts_by_tp
from ast import literal_eval
import numpy as np
from nonkgmethod import find_joinable
import math
import copy
from kgmethod import kgscore
from nonkgmethod import nonkgscore

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

def get_entropy(score_lst : list):
    total = 0.0
    score_sum = sum(score_lst)
    
    for s in score_lst:
        prob = s / score_sum
        logprob = math.log(prob, 2)
        total += prob * logprob
    
    total = -total
    
    return total

def insert_el(lst : list, el, elname, topk):
    #the easiest, and worst way to do this is to
    #just insert the element, sort the list again,
    #take the top k, and return.
    res = lst + [(elname[0], elname[1], el)]
    return sorted(res, key=lambda x: x[2], reverse=True)[:topk]

def run_full(infile, lake_dir, lsh_ind, ent_thresh=0.5, has_gt=False, topk=3):
    topk_kg = []
    topk_nonkg = []
    #first, find all joinable tables to the input table
    lake_joins = find_joinable(infile, lsh_ind, lake_dir)
    #first, get the classification for the infile
    i_cpl = sem_classify(infile, is_test=has_gt)
    print("table class/prop pairs: {}, {}".format(infile, i_cpl))
    #sample of cplabels--
    #{'<http://dbpedia.org/property/annualRidership>': [(1.0, "(['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')")], '<http://dbpedia.org/ontology/numberOfLines>': [(1.0, "(['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')")]}
    #now, for each table in the data lake
    #first, get the list of class/property label distributions
    in_entlst = []
    for cname in i_cpl:
        in_labels = i_cpl[cname]
        in_scores = [e[0] for e in in_labels]
        in_entlst.append(get_entropy(in_scores))
    
    for intup in lake_joins:
        for otup in lake_joins[intup]:
            o_cpl = sem_classify(otup[0], is_test=has_gt)
            print("table class/prop pairs: {}, {}".format(otup[0], o_cpl))
            entlst = []
            for cname in o_cpl:
                o_labels = o_cpl[cname]
                o_scores = [e[0] for e in o_labels]
                entlst.append(get_entropy(o_scores))
    
            entlst += in_entlst
            avg_ent = sum(entlst) / len(entlst)
            if avg_ent < ent_thresh:
                #then, get the kgscore
                indf = pd.read_csv(infile)
                outdf = pd.read_csv(otup[0])
                injk = intup[1]
                outjk = otup[1]
                
                kgsc, kgrels = kgscore(indf, outdf, injk, outjk, i_cpl, o_cpl, intup[0], otup[0])
                #sequential insertion
                topk_kg = insert_el(topk_kg, kgsc, otup, topk)
            else:
                injk = intup[1]
                outjk = otup[1]
                nonkgsc = nonkgscore(infile, otup[0], injk, outjk, lake_dir, lsh_ind, 'all_lake_fts.json')
                topk_nonkg = insert_el(topk_nonkg, nonkgsc, otup, topk)
    
    with open('fullresults_kg.txt', 'w+') as fh:
        print(topk_kg, file=fh)
    
    with open('fullresults_nonkg.txt', 'w+') as fh:
        print(topk_nonkg, file=fh)

#def display_results():
    
    
    

if __name__ == "__main__":
    #test classification using models on properties we already have
    # print(sem_classify('busridertbl.csv', is_test=True))
    #try running full method
    run_full('demo_lake/busridertbl.csv', 'demo_lake', 'all_lake_joins.json', has_gt=True)
    
    

