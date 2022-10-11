from fuzzycmeans import FCM
from SPARQLWrapper import SPARQLWrapper, JSON
from semhelper import check_type, get_fts_by_tp
from ast import literal_eval
import sys
import pickle
import os

"""
Purpose: Semantically Label all the columns in the data lake and
store their membership vectors.
"""

def parse_dt(b_ent):
    if 'datatype' in b_ent:
        return b_ent['datatype']
    else:
        return ''

def convert_val(b_ent):
    dt = b_ent['datatype']
    if 'integer' in dt.lower():
        return int(b_ent['value'])
    
    return float(b_ent['value'])
        

def nums_query(cp):
    query = 'select ?val where { '
    classes = cp[0]
    prop = cp[1]
    for cl in classes:
        query += '?subject a ' + cl + ' . '
    
    query += '?subject ' + prop + ' ?val . } LIMIT 10000'
    print(query)
    qname = '_'.join(classes)
    if os.sep in prop:
        last_name = prop.split(os.sep)[-1]
        qname += last_name
    
    if os.path.exists('query_answers/' + qname + '.json'):
        print("Reading from Existing")
        with open('query_answers/' + qname + '.json', 'r') as fh:
            st = fh.read()
            results = literal_eval(st)
    else:
        print("Querying Endpoint")
        sparql = SPARQLWrapper("https://dbpedia.org/sparql/", agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        with open('query_answers/' + qname + '.json', 'w+') as fh:
            print(results, file=fh)
            
    valdt = ''
    val_lst = []
    for b in results['results']['bindings']:
        if valdt == '':
            valdt = parse_dt(b['val'])
            if valdt == '':
                continue
        
        cur_dt = parse_dt(b['val'])
        if cur_dt != valdt:
            continue
        bval = convert_val(b['val'])
        val_lst.append(bval)
    
    
    return val_lst
        
        

#given a list of numeric knowledge graph class-property pairs,
#where the class is a list of dbpedia class names,
# and the property is the name of a dbpedia property of an entity of the class,
#construct and return the numeric features for each of these pairs
def construct_numfts(cp_pairs):
    cp2fts = {}
    
    for cp in cp_pairs:
        cp_vals = nums_query(cp)
        num_tp = check_type(cp_vals)
        num_fts = get_fts_by_tp(cp_vals, num_tp)
        cp2fts[str(cp)] = num_fts
    
    with open('kgprop_numfts.json', 'w+') as fh:
        print(cp2fts, file=fh)

#given a dictionary of class/property pairs to numeric features on disk,
#create and store the fuzzy c-means clustering models on disk.
def create_fcm():
    with open('kgprop_numfts.json', 'r') as fh:
        st = fh.read()
        cp2fts = literal_eval(st)
    
    centroids = {}
    centroid_names = {}
    
    #maximum number of categories out of any categorical class-property pair...
    mx_cats = sys.float_info.min
    for cp in cp2fts:
        fts = cp2fts[cp]
        for tp in fts:
            if tp == 'categorical':
                #...let's trust the actual length of the distribution
                cat_len = len(fts[tp][1])
                if cat_len > mx_cats:
                    mx_cats = cat_len
    
    #now, cluster
    for cp in cp2fts:
        fts = cp2fts[cp]
        print("fts: {}".format(fts))
        #fts should only have one key, which is the numeric type
        tp = list(fts.keys())[0]
        if tp == 'categorical':
            if len(fts[tp][1]) < mx_cats:
                #pad with zeros
                new_hist = fts[tp][1] + [0] * (mx_cats - len(fts[tp][[1]]))
            else:
                new_hist = fts[tp][1]
            
            new_fts = [fts[tp][0]] + new_hist
        else:
            new_fts = list(fts[tp])
        
        if tp in centroids and tp in centroid_names:
            centroids[tp].append(new_fts)
            centroid_names[tp].append(cp)
        else:
            centroids[tp] = [new_fts]
            centroid_names[tp] = [cp]
    
    #create one cluster per numeric type
    for tp in centroids:
        fcm = FCM(n_clusters=len(centroids[tp]))
        print(centroids[tp])
        fcm.fit(centroids[tp], range(len(centroids[tp])))
    
        with open('fcm_' + tp + 'model.pkl', 'wb') as fh:
            pickle.dump(fcm, fh)
        
        with open('fcm_' + tp + 'centroids.txt', 'w+') as fh:
            print(centroid_names[tp], file=fh)
        
        with open('max_numfts.txt', 'w+') as fh:
            print(mx_cats, file=fh)

if __name__ == "__main__":
    init_pairs = [(['dbo:BusCompany'], '<http://dbpedia.org/property/annualRidership>'), 
                  (['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')]
    construct_numfts(init_pairs)
    create_fcm()
    
