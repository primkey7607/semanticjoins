import os
from semhelper import check_type, get_fts_by_tp
import pandas as pd
from ast import literal_eval
import numpy as np
import pickle
from find_lsh import query_lsh

"""
Purpose: Run and index the non-KG query results. The most expensive part of this
is computing minhashes on the input column, our identified proxy table,
and the output data lake table.

Design:
1. Find all tables that join with the output table (not the input table).
2. Of those, find the table most similar to the input by whichever table has the
most similar column to one of the input table's columns. Ideally,
we'd actually want to find the table that has the most similar distributions
to the input.
3. The cardinality of the join of this proxy table with the output table is the
score.
"""

def gen_lakefts(lake_dir, fts_name):
    fc2fts = {}
    for f in os.listdir(lake_dir):
        fname = os.path.join(lake_dir, f)
        fc2fts[fname] = {}
        newdf = pd.read_csv(fname)
        for c in newdf.columns:
            ctp = check_type(newdf[c].to_list())
            cfts = get_fts_by_tp(newdf[c].to_list(), ctp)
            fc2fts[fname][c] = cfts
    
    with open(fts_name, 'w+') as fh:
        print(fc2fts, file=fh)

def parse_fckey(st):
    #this, we can reverse engineer.
    init_ind = st.index('.csv')
    fname = st[:init_ind + 4]
    fpt2 = fname.replace(os.sep, '_')
    all_flen = len(fname + '_' + fpt2 + '_')
    cname = st[all_flen + 1:]
    return fname, cname
    
    

def parse_fcval(st, lake_dir):
    #this, i'm not so sure can be reverse engineered...
    #because, how do I know where the os.seps are, as opposed to underscores?
    #for now, i can just take advantage of the fact that we need to know
    #the data lake directory
    dir_sepd = lake_dir.replace(os.sep, '_')
    dir_ind = st.index(dir_sepd)
    f_start = dir_ind + len(dir_sepd) + 1
    f_end = st.index('.csv') + 4
    fname = st[f_start:f_end]
    
    full_fname = lake_dir + os.sep + fname
    cname = st[f_end + 1:]
    
    return full_fname, cname

#given the name of a table in the data lake,
#and the name of the json file containing a dictionary with all joins
#in the data lake,
#find all joinable tables to the given one in the data lake
def find_joinable(outname, ind_name, lake_dir):
    #query the minhash lsh index on the data lake
    #to get a dictionary whose keys are file/column names of outname
    #and whose values are lists of file/column pairs.
    prox_dct = {}
    if not os.path.exists(ind_name):
        query_lsh(outname, 'mh_dict.json', 'lake_lshind.pkl', ind_name)
    
    with open(ind_name, 'r') as fh:
        st = fh.read()
        dct = literal_eval(st)
    
    for k in dct:
        outf, outc = parse_fckey(k)
        if outf == outname:
            #then, get the list of similar fc pairs 
            for ent in dct[k]:
                prox_f, prox_c = parse_fcval(ent, lake_dir)
                #avoid any self joins
                if prox_f == outname:
                    continue
                if (outf, outc) in prox_dct:
                    prox_dct[(outf, outc)].append((prox_f, prox_c))
                else:
                    prox_dct[(outf, outc)] = [(prox_f, prox_c)]
    
    return prox_dct

def find_proxy(indf, out_joins, lake_dir, fts_name, thresh=0.05):
    if not os.path.exists(fts_name):
        gen_lakefts(lake_dir, fts_name)
    
    with open(fts_name, 'r') as fh:
        st = fh.read()
        fc2fts = literal_eval(st)
    
    indf_fts = {}
    for c in indf.columns:
        ctp = check_type(indf[c].to_list())
        cfts = get_fts_by_tp(indf[c].to_list(), ctp)
        indf_fts[c] = cfts
    
    f_sims = {}
    relevant_fset = set()
    for intup in out_joins:
        for tup in out_joins[intup]:
            fname = tup[0]
            fk = tup[1]
            relevant_fset.add(fname)
    
    relevant_fs = list(relevant_fset)
    
    for fname in relevant_fs:
        for cname in fc2fts[fname]:
            out_fts = fc2fts[fname][cname]
            outtp = list(out_fts.keys())[0]
            for inc in indf_fts:
                incfts = indf_fts[inc]
                intp = list(incfts.keys())
                if intp == outtp:
                    invec = np.array(incfts[intp])
                    outvec = np.array(out_fts[outtp])
                    #scale to between 0 and 1
                    innorm = invec / np.linalg.norm(invec)
                    outnorm = outvec / np.linalg.norm(outvec)
                    if np.linalg.norm(outnorm - innorm) < thresh:
                        if fname in f_sims:
                            f_sims[fname] += 1
                        else:
                            f_sims[fname] = 1
        
    
    max_sim = max([f_sims[fname] for f in f_sims])
    max_fname = None
    
    #pick the first file name with the highest similarity
    for fname in f_sims:
        if f_sims[fname] == max_sim:
            max_fname = fname
            break
    fks = []
    
    #now, get the input fks, and the fks for the most similar table
    for tup in out_joins:
        infname = tup[0]
        incname = tup[1]
        for otup in out_joins[tup]:
            if otup[0] == max_fname:
                cname = otup[1]
                fks.append((incname, cname))
    
    return max_fname, fks

def find_bestcard(tbl, fks, outfname):
    df1 = pd.read_csv(tbl)
    df2 = pd.read_csv(outfname)
    card_bound = df1.shape[0] * df2.shape[0]
    #try out the different joins
    max_card = -1
    max_ent = None
    for fk in fks:
        jk1 = fk[0] #the output table's FK
        jk2 = fk[1] #the proxy table's FK
        joindf = df1.merge(df2, left_on=jk2, right_on=jk1)
        joincard = joindf.shape[0] / card_bound
        if joincard > max_card:
            max_card = joincard
            max_ent = (jk1, jk2)
    
    return max_ent, max_card

def nonkgscore(infname, outfname, jk1, jk2, lake_dir, lsh_ind, fts_name):
    df1 = pd.read_csv(infname)
    
    #find all tables joinable to the output table first
    out_joins = find_joinable(outfname, lsh_ind, lake_dir)
    
    #find the joinable table with the most matching numeric distributions
    #to the input
    proxy_table, proxy_fks = find_proxy(df1, out_joins, lake_dir, fts_name)
    best_fks, cardinality = find_bestcard(proxy_table, proxy_fks, outfname)
    print("Best Foreign Keys: {}".format(best_fks))
    
    return cardinality
    
    
    
    
    
    