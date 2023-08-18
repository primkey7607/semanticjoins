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
import seaborn as sns
from IPython.display import display_html, display
from itertools import chain,cycle
import time

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
            if coltp == 'count':
                coltp = 'counts'
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

def insert_el(lst : list, el, elname, infile, injk, topk, proxies=None):
    #the easiest, and worst way to do this is to
    #just insert the element, sort the list again,
    #take the top k, and return.
    if proxies != None:
        res = lst + [(elname[0], elname[1], infile, injk, proxies[0], proxies[1], el)]
        return sorted(res, key=lambda x: x[6], reverse=True)[:topk]
    res = lst + [(elname[0], elname[1], infile, injk, el)]
    return sorted(res, key=lambda x: x[4], reverse=True)[:topk]

def run_full(infile, lake_dir, lsh_ind, ent_thresh=0.5, has_gt=False, topk=3, test_nkg=False):
    topk_kg = []
    topk_nonkg = []
    #first, find all joinable tables to the input table
    with open('all_lakes_joined_parsed.json', 'r') as fh:
        st = fh.read()
        lake_joins = literal_eval(st)
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
            if avg_ent < ent_thresh and not test_nkg:
                #then, get the kgscore
                indf = pd.read_csv(infile)
                outdf = pd.read_csv(otup[0])
                injk = intup[1]
                outjk = otup[1]
                
                kgsc, kgrels = kgscore(indf, outdf, injk, outjk, i_cpl, o_cpl, intup[0], otup[0])
                #sequential insertion
                topk_kg = insert_el(topk_kg, kgsc, otup, infile, injk, topk, kg_rels=kgrels)
            # else:
            #     injk = intup[1]
            #     outjk = otup[1]
            #     proxy_info, nonkgsc = nonkgscore(infile, otup[0], injk, outjk, lake_dir, lsh_ind, 'all_lake_fts')
            #     topk_nonkg = insert_el(topk_nonkg, nonkgsc, otup, infile, injk, topk, proxies=proxy_info)
    
    if topk_kg != []:
        with open(infile[:-4] + '_fullresults_kg.txt', 'w+') as fh:
            print({ infile : topk_kg}, file=fh)
    else:
        print("topk_kg results empty!!")
    
    if topk_nonkg != []:
        with open(infile[:-4] + '_fullresults_nonkg.txt', 'w+') as fh:
            print({ infile : topk_nonkg}, file=fh)
    else:
        print("topk_nonkg results empty!!")

# def display_results(inp_tbl):
#     cm = sns.light_palette("green", as_cmap=True)
#     # Set CSS properties for th elements in dataframe
#     th_props = [
#       ('font-size', '11px'),
#       ('text-align', 'center'),
#       ('font-weight', 'bold'),
#       ('color', '#6d6d6d'),
#       ('background-color', '#f7f7f9')
#       ]
    
#     # Set CSS properties for td elements in dataframe
#     td_props = [
#       ('font-size', '11px')
#       ]
    
#     # Set table styles
#     styles = [
#       dict(selector="th", props=th_props),
#       dict(selector="td", props=td_props)
#       ]
    
#     return (inp_tbl.style
#    .set_properties(**{'background-color' : 'green'}, subset=['dbo:BusCompany'])
#    .set_properties(**{'background-color' : 'yellow'}, subset=['dbo:regionServed'])
#   #.background_gradient(cmap=cm, subset=['dbo:BusCompany','dbo:regionServed'])
#   #.highlight_max(subset=['dbo:BusCompany','dbo:regionServed'])
#   .set_caption('The ground truth is in green, and the join key is yellow.')
#   #.format({'dbo:regionServed': "{:.2%}"})
#   .set_table_styles(styles))

"""
Display Intuition: for every table we have in top-k: if non-KG, then display the input table,
that table, and proxy table side-by-side, with the join key highlighted between the tables.
Else if KG, then display the input table with the KG entities highlighted and the join key highlighted
with a different color. We can make this happen just by putting the dataframes side-by-side,
but the final thing we would need is the ability to "title" these pairs with
a relationship score, and score name, if KG.
"""

def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

def display_df(df, col2highlight, styles):
    return (df.style
   #.set_properties(**{'background-color' : 'green'}, subset=['dbo:BusCompany'])
   .set_properties(**{'background-color' : 'yellow'}, subset=[col2highlight])
  #.background_gradient(cmap=cm, subset=['dbo:BusCompany','dbo:regionServed'])
  #.highlight_max(subset=['dbo:BusCompany','dbo:regionServed'])
  #.set_caption('The ground truth is in green, and the join key is yellow.')
  #.format({'dbo:regionServed': "{:.2%}"})
  .set_table_styles(styles))

#specifically intended for our column names,
#and our string column values. 
def prettify_st(st : str):
    if st.startswith('dbo:'):
        new_st = st[4:]
        return new_st
    elif 'http' in st:
        new_st = st.split('/')[-1]
        if '>' in new_st:
            new_st = new_st.replace('>', '')
        return new_st
    else:
        #leave it alone
        return st
        

# display the input table with highlighted input column and entity columns
# alongside the output table
# and add a title to the whole affair with a relationship strength. 
# (we don't know how to do this yet)
def display_kg(indf, in_ent, in_jk, df, ent_col, jk_col, styles, title, in_title, rel_score):
    if 'Unnamed: 0.1' in indf.columns:
        indf = indf.drop(columns=['Unnamed: 0.1'])
    if 'Unnamed: 0.1' in df.columns:
        df = df.drop(columns=['Unnamed: 0.1'])
    
    indf[in_jk + '_id'] = indf[in_jk].astype('category').cat.rename_categories(range(1, indf[in_jk].nunique()+1))
    if indf.dtypes[in_jk] == 'object':
        indf = indf.drop(columns=[in_jk]) #get rid of the string version of the join key--it just distracts from our point.
    df[jk_col + '_id'] = df[jk_col].astype('category').cat.rename_categories(range(1, df[jk_col].nunique()+1))
    
    pretty_incol_lst = [{incol : prettify_st(incol)} for incol in indf.columns]
    pretty_incols = {}
    for e in pretty_incol_lst:
        for k in e:
            pretty_incols[k] = e[k]
    
    pretty_outcol_lst = [{outcol : prettify_st(outcol)} for outcol in df.columns]
    pretty_outcols = {}
    for e in pretty_outcol_lst:
        for k in e:
            pretty_outcols[k] = e[k]
    
    pretty_indf = indf.rename(columns=pretty_incols)
    pretty_df = df.rename(columns=pretty_outcols)
    # print(pretty_indf.columns)
    # print(pretty_df.columns)
    # print(pretty_indf.index.is_unique)
    # print(pretty_df.index.is_unique)
    
    inobj_cols = pretty_indf.select_dtypes(include='object').head()
    oobj_cols = pretty_df.select_dtypes(include='object').head()
    
    infl_cols = pretty_indf.select_dtypes(include='float').head()
    ofl_cols = pretty_df.select_dtypes(include='float').head()
    
    
    for c in inobj_cols:
        pretty_indf[c] = pretty_indf[c].apply(lambda x: prettify_st(x))
    
    for c in oobj_cols:
        pretty_df[c] = pretty_df[c].apply(lambda x: prettify_st(x))
    
    infl_dct = {}
    ofl_dct = {}
    for c in infl_cols:
        infl_dct[c] = '{:.2f}'
    
    for c in ofl_cols:
        ofl_dct[c] = '{:.2f}'
    
    
    
    pd.set_option('display.max_colwidth', 10)
    #space = "\xa0" * 10
    space = ""
    indf_styler = (pretty_indf.style
                   .set_table_attributes("style='display:inline; margin-right:20px;'")
                   .set_properties(**{'background-color' : 'yellow'}, subset=[pretty_incols[in_jk + '_id']])
                   .set_properties(**{'background-color' : 'lightgreen'}, subset=[pretty_incols[in_ent]])
                   .set_caption('Input Table: ' + in_title)
                   #.set_table_styles(styles)
                   .format(infl_dct))
    
    
    
    
    outdf_styler = (pretty_df.style
                    .set_table_attributes("style='display:inline'")
                    .set_properties(**{'background-color' : 'yellow'}, subset=[pretty_outcols[jk_col + '_id']])
                    .set_properties(**{'background-color' : 'lightgreen'}, subset=[pretty_outcols[jk_col]])
                    .set_caption('KG Join Table: ' + title + '\nRelationship Strength: ' + str(rel_score))
                    #.set_table_styles(styles)
                    .format(ofl_dct))
    
    final_display_obj = indf_styler._repr_html_() + outdf_styler._repr_html_()
    final_display_obj = final_display_obj.replace('table','table style="display:inline"')
    return final_display_obj


def display_nonkg(indf, in_jk, outdf, proxy_df, jk_col, proxy_col, styles, title, proxy_title, in_title, rel_score):
    if 'Unnamed: 0.1' in indf.columns:
        indf = indf.drop(columns=['Unnamed: 0.1'])
    if 'Unnamed: 0.1' in outdf.columns:
        outdf = outdf.drop(columns=['Unnamed: 0.1'])
    if 'Unnamed: 0.1' in proxy_df.columns:
        proxy_df = proxy_df.drop(columns=['Unnamed: 0.1'])
    
    if prettify_st(in_jk) == in_jk:
        in_jk_id = in_jk
    else:
        in_jk_id = in_jk + '_id'
    
    if prettify_st(jk_col) == jk_col:
        jk_col_id = jk_col
    else:
        jk_col_id = jk_col + '_id'
    
    if prettify_st(proxy_col) == proxy_col:
        p_col_id = proxy_col
    else:
        p_col_id = proxy_col + '_id'
    
    if in_jk_id != in_jk:
        indf[in_jk + '_id'] = indf[in_jk].astype('category').cat.rename_categories(range(1, indf[in_jk].nunique()+1))
    
    if jk_col_id != jk_col:
        outdf[jk_col + '_id'] = outdf[jk_col].astype('category').cat.rename_categories(range(1, outdf[jk_col].nunique()+1))
    
    if p_col_id != proxy_col:
        proxy_df[proxy_col + '_id'] = proxy_df[proxy_col].astype('category').cat.rename_categories(range(1, proxy_df[proxy_col].nunique()+1))
    
    pretty_incol_lst = [{incol : prettify_st(incol)} for incol in indf.columns]
    pretty_incols = {}
    for e in pretty_incol_lst:
        for k in e:
            pretty_incols[k] = e[k]
    
    pretty_outcol_lst = [{outcol : prettify_st(outcol)} for outcol in outdf.columns]
    pretty_outcols = {}
    for e in pretty_outcol_lst:
        for k in e:
            pretty_outcols[k] = e[k]
    
    pretty_proxcol_lst = [{proxcol : prettify_st(proxcol)} for proxcol in proxy_df.columns]
    pretty_proxcols = {}
    for e in pretty_proxcol_lst:
        for k in e:
            pretty_proxcols[k] = e[k]
    
    pretty_indf = indf.rename(columns=pretty_incols)
    pretty_df = outdf.rename(columns=pretty_outcols)
    pretty_prox = proxy_df.rename(columns=pretty_proxcols)
    # print(pretty_indf.columns)
    # print(pretty_df.columns)
    # print(pretty_indf.index.is_unique)
    # print(pretty_df.index.is_unique)
    
    inobj_cols = pretty_indf.select_dtypes(include='object').head()
    oobj_cols = pretty_df.select_dtypes(include='object').head()
    pobj_cols = pretty_prox.select_dtypes(include='object').head()
    
    infl_cols = pretty_indf.select_dtypes(include='float').head()
    ofl_cols = pretty_df.select_dtypes(include='float').head()
    pfl_cols = pretty_prox.select_dtypes(include='float').head()
    
    
    for c in inobj_cols:
        pretty_indf[c] = pretty_indf[c].apply(lambda x: prettify_st(x))
    
    for c in oobj_cols:
        pretty_df[c] = pretty_df[c].apply(lambda x: prettify_st(x))
    
    for c in pobj_cols:
        pretty_prox[c] = pretty_prox[c].apply(lambda x: prettify_st(x))
    
    infl_dct = {}
    ofl_dct = {}
    pfl_dct = {}
    for c in infl_cols:
        infl_dct[c] = '{:.2f}'
    
    for c in ofl_cols:
        ofl_dct[c] = '{:.2f}'
    
    for c in pfl_cols:
        pfl_dct[c] = '{:.2f}'
    
    
    
    pd.set_option('display.max_colwidth', 10)
    #space = "\xa0" * 10
    space = ""
    if prettify_st(in_jk) == in_jk:
        in_jk_id = in_jk
    else:
        in_jk_id = in_jk + '_id'
    
    indf_styler = (pretty_indf.style
                   .set_table_attributes("style='display:inline; margin-right:20px;'")
                   .set_properties(**{'background-color' : 'yellow'}, subset=[pretty_incols[in_jk_id]])
                   #.set_properties(**{'background-color' : 'green'}, subset=[pretty_incols[in_ent]])
                   .set_caption('Input Table: ' + in_title)
                   #.set_table_styles(styles)
                   .format(infl_dct))
    
    if prettify_st(jk_col) == jk_col:
        jk_col_id = jk_col
    else:
        jk_col_id = jk_col + '_id'
    
    outdf_styler = (pretty_df.style
                    .set_table_attributes("style='display:inline'")
                    .set_properties(**{'background-color' : 'yellow'}, subset=[pretty_outcols[jk_col_id]])
                    .set_caption('Non-KG Join Table: ' + title + '\nRelationship Strength: ' + str(rel_score))
                    #.set_table_styles(styles)
                    .format(ofl_dct))
    
    if prettify_st(proxy_col) == proxy_col:
        p_col_id = proxy_col
    else:
        p_col_id = proxy_col + '_id'
    
    proxdf_styler = (pretty_prox.style
                    .set_table_attributes("style='display:inline'")
                    .set_properties(**{'background-color' : 'yellow'}, subset=[pretty_proxcols[p_col_id]])
                    .set_caption('Non-KG Proxy Table: ' + proxy_title)
                    #.set_table_styles(styles)
                    .format(pfl_dct))
    
    final_display_obj = indf_styler._repr_html_() + proxdf_styler._repr_html_() + outdf_styler._repr_html_()
    final_display_obj = final_display_obj.replace('table','table style="display:inline"')
    return final_display_obj

def display_usecases():
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv('all_usecases.csv')
    return df

def display_results(infile):
    ent_lst = ['dbo:BusCompany', 'dbo:Hospital', 'dbo:Politician',
               'dbo:SoccerPlayer', 'dbo:Bank']
    #12/6 TODO: add the input table here as well.
    #the results are now dictionaries instead of lists,
    #where key = input file name, and value = list of KG/non-KG tuples of tables, columns, and scores.
    #TODO: store all the results in one file. For now, we'll just use a condition
    
    if infile == 'demo_lake/busridertbl.csv':
        with open('fullresults_kg.txt', 'r') as fh:
            st = fh.read()
            kg_res = literal_eval(st)
        
        # with open('fullresults_nonkg.txt', 'r') as fh:
        #     st = fh.read()
        #     nonkg_res = literal_eval(st)
        nonkg_res = []
    elif infile == 'demo_lake/ETF prices.csv':
        with open('stockresults_kg.txt', 'r') as fh:
            st = fh.read()
            kg_res = literal_eval(st)
        
        with open('stockresults_nonkg.txt', 'r') as fh:
            st = fh.read()
            nonkg_res = literal_eval(st)
    elif infile == 'demo_lake/soccertbl.csv':
        with open('soccerresults_kg.txt', 'r') as fh:
            st = fh.read()
            kg_res = literal_eval(st)
        
        with open('soccerresults_nonkg.txt', 'r') as fh:
            st = fh.read()
            nonkg_res = literal_eval(st)
    
    indf = pd.read_csv(infile, nrows=5)
    in_ent = ''
    for c in indf.columns:
        if c in ent_lst:
            in_ent = c
    
    # Set colormap equal to seaborns light green color palette
    cm = sns.light_palette("green", as_cmap=True)
    
    # Set CSS properties for th elements in dataframe
    th_props = [
      ('font-size', '11px'),
      ('text-align', 'center'),
      ('font-weight', 'bold'),
      ('color', '#6d6d6d'),
      ('background-color', '#f7f7f9')
      ]
    
    # Set CSS properties for td elements in dataframe
    td_props = [
      ('font-size', '11px')
      ]
    
    # Set table styles
    styles = [
      dict(selector="th", props=th_props),
      dict(selector="td", props=td_props)
      ]
    
    #first, display the knowledge graph results
    for r in kg_res:
        in_jk = r[1]
        outdf = pd.read_csv(r[2], nrows=5)
        jk_col = r[3]
        ent_col = in_ent
        
        #TODO: right now, this will color the same column both yellow and green.
        #I think the more correct way to do this is to have a separate ID
        #column for entities in both tables, and color these yellow (as the join key)
        #the only problem is that we would have to assume that IDs correspond to entity
        #names exactly, and that's something you see in DBMSs, not data lakes.
        #And if you don't see this in data lakes, and you only see the numeric features,
        #then the question is--are we saying we'll find joins that don't exist?
        #if so, that's a whole new problem! Anyway, let's not think too much about that for now.
        #tbl_displays = display_kg(outdf, ent_col, jk_col, styles, r[0])
        tbl_displays = display_kg(indf, in_ent, in_jk, outdf, ent_col, jk_col, styles, r[2], infile, r[-1])
        display_html(tbl_displays, raw=True)
    
    for r in nonkg_res:
        outdf = pd.read_csv(r[0], nrows=5)
        jk_col = r[1]
        proxy_df = pd.read_csv(r[2], nrows=5)
        proxy_fk = r[3][0][1]
        in_jk = r[3][0][0]
        tbl_displays = display_nonkg(indf, in_jk, outdf, proxy_df, jk_col, proxy_fk, styles, r[0], r[2], infile, r[-1])
        display_html(tbl_displays, raw=True)
        
    
    

if __name__ == "__main__":
    #test classification using models on properties we already have
    # print(sem_classify('busridertbl.csv', is_test=True))
    #try running full method
    results = run_full('demo_lake/busridertbl.csv', 'demo_lake', 'all_lake_joins.json', has_gt=True, test_nkg=True)
    print(results)
    #let's test displays
    # indf = pd.read_csv('demo_lake/busridertbl.csv')
    # in_jk = 'dbo:regionServed'
    # in_ent = 'dbo:BusCompany'
    # outdf = pd.read_csv('demo_lake/busriderjoin.csv')
    # out_jk = 'dbo:regionServed'
    # out_ent = 'dbo:regionServed'
    # rel_score = 0.04950495049504951
    # title = 'demo_lake/busriderjoin.csv'
    
    # # Set colormap equal to seaborns light green color palette
    # cm = sns.light_palette("green", as_cmap=True)
    
    # # Set CSS properties for th elements in dataframe
    # th_props = [
    #   ('font-size', '11px'),
    #   ('text-align', 'center'),
    #   ('font-weight', 'bold'),
    #   ('color', '#6d6d6d'),
    #   ('background-color', '#f7f7f9')
    #   ]
    
    # # Set CSS properties for td elements in dataframe
    # td_props = [
    #   ('font-size', '11px')
    #   ]
    
    # # Set table styles
    # styles = [
    #   dict(selector="th", props=th_props),
    #   dict(selector="td", props=td_props)
    #   ]
    
    # display_kg(indf, in_ent, in_jk, outdf, out_ent, out_jk, styles, title, rel_score)
    
    

