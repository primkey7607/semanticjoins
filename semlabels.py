from fuzzycmeans import FCM
from SPARQLWrapper import SPARQLWrapper, JSON
from semhelper import check_type, get_fts_by_tp
from ast import literal_eval
import sys
import pickle
import os
import pandas as pd

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
    if b_ent['type'] == 'literal' and 'xml:lang' in b_ent:
        return None
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

def allprop_query(cp_pairs):
    subs = ''
    body = ''
    outdct = {}
    
    #first, make a dictionary from unique classes to lists of their properties
    cl2props = {}
    sub2cl = {}
    pname2prop = {}
    
    for cp in cp_pairs:
        cl = tuple(cp[0])
        prop = cp[1]
        if cl in cl2props:
            cl2props[cl].append(prop)
        else:
            cl2props[cl] = [prop]
    
    p_ind = 0
    for j,cl in enumerate(cl2props):
        sname = '?subject' + str(j)
        sub2cl[sname[1:]] = cl
        subs += sname + ', '
        
        for cln in cl:
            body += sname + ' a ' + cln + ' . '
        
        for i,p in enumerate(cl2props[cl]):
            pname = '?prop' + str(p_ind)
            p_ind += 1
            if j == len(cl2props) - 1 and i == len(cl2props[cl]) - 1:
                subs += pname + ' '
            else:
                subs += pname + ', '
            pname2prop[pname[1:]] = p
            body += sname + ' ' + p + ' ' + pname + ' . '
        
    
    query = 'select ' + subs + ' { ' + body + ' } LIMIT 100'
    print(query)
    sparql = SPARQLWrapper("https://dbpedia.org/sparql/", agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    vnames = results['head']['vars']
    
    for b in results['results']['bindings']:
        for v in vnames:
            if 'subject' in v:
                clname = '_'.join(sub2cl[v])
                if clname in outdct:
                    outdct[clname].append(b[v]['value'])
                else:
                    outdct[clname] = [b[v]['value']]
            else:
                prop = pname2prop[v]
                bval = convert_val(b[v])
                if prop in outdct:
                    outdct[prop].append(bval)
                else:
                    outdct[prop] = [bval]
    
    return outdct

def join_query(cp_pairs : list, jprops : dict, rels : list):
    subs = ''
    body = ''
    outdct = {}
    joindct = {}
    
    #first, make a dictionary from unique classes to lists of their properties
    cl2props = {}
    sub2cl = {}
    pname2prop = {}
    join2rel = {}
    jpname2prop = {}
    
    
    for cp in cp_pairs:
        cl = tuple(cp[0])
        prop = cp[1]
        if cl in cl2props:
            cl2props[cl].append(prop)
        else:
            cl2props[cl] = [prop]
    
    p_ind = 0
    for j,cl in enumerate(cl2props):
        sname = '?subject' + str(j)
        sub2cl[sname[1:]] = cl
        subs += sname + ', '
        
        for cln in cl:
            body += sname + ' a ' + cln + ' . '
        
        for i,p in enumerate(cl2props[cl]):
            pname = '?prop' + str(p_ind)
            p_ind += 1
            subs += pname + ', '
            pname2prop[pname[1:]] = p
            body += sname + ' ' + p + ' ' + pname + ' . '
    
    #now, we also want to include the join values in the query
    jp_ind = 0
    for i,r in enumerate(rels):
        cl = tuple(r[0])
        #this should only have one element
        subname = [s for s in sub2cl if sub2cl[s] == cl][0]
        subname = '?' + subname
        rel = r[1]
        jproplst = jprops[rel]
        
        jname = '?join' + str(i)
        join2rel[jname[1:]] = rel
        subs += jname + ', '
        body += subname + ' ' + rel + ' ' + jname + ' . '
        
        for j,jp in enumerate(jproplst):
            jpname = '?jprop' + str(jp_ind)
            jp_ind += 1
            if j == len(jproplst) - 1 and i == len(rels) - 1:
                subs += jpname + ' '
            else:
                subs += jpname + ', '
            jpname2prop[jpname[1:]] = jp
            body += jname + ' ' + jp + ' ' + jpname + ' . '
    
    query = 'select ' + subs + ' { ' + body + ' } LIMIT 100'
    print(query)
    sparql = SPARQLWrapper("https://dbpedia.org/sparql/", agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    vnames = results['head']['vars']
    
    for b in results['results']['bindings']:
        for v in vnames:
            if 'subject' in v:
                clname = '_'.join(sub2cl[v])
                if clname in outdct:
                    outdct[clname].append(b[v]['value'])
                else:
                    outdct[clname] = [b[v]['value']]
            elif 'join' in v: #this is our join key
                jkname = join2rel[v]
                if jkname in outdct:
                    outdct[jkname].append(b[v]['value'])
                else:
                    outdct[jkname] = [b[v]['value']]
                
                if jkname in joindct:
                    joindct[jkname].append(b[v]['value'])
                else:
                    joindct[jkname] = [b[v]['value']]
            
            elif 'jprop' in v:
                jpname = jpname2prop[v]
                bval = convert_val(b[v])
                if jpname in joindct:
                    joindct[jpname].append(bval)
                else:
                    joindct[jpname] = [bval]
            elif 'prop' in v:
                prop = pname2prop[v]
                bval = convert_val(b[v])
                if prop in outdct:
                    outdct[prop].append(bval)
                else:
                    outdct[prop] = [bval]
    
    return outdct, joindct
    

#for testing purposes, construct a table from KG class instances
def tbls_from_kg(cp_pairs, outname):
    dfdct = allprop_query(cp_pairs)
    
    newdf = pd.DataFrame(dfdct)
    newdf.to_csv(outname)

def joins_from_kg(cp_pairs, join_pairs, rels, outname, outjoin):
    dfdct, joindct = join_query(cp_pairs, join_pairs, rels)
    
    newdf = pd.DataFrame(dfdct)
    newjoin = pd.DataFrame(joindct)
    
    newdf.to_csv(outname)
    
    newjoin.to_csv(outjoin)
        
if __name__ == "__main__":
    #we will first generate all the KG tables here
    # bus_pairs = [(['dbo:BusCompany'], '<http://dbpedia.org/property/annualRidership>'), 
    #               (['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')]
    # #construct_numfts(init_pairs)
    # #create_fcm()
    # #tbls_from_kg(bus_pairs, 'demo_lake/busridertbl.csv')
    
    # bus_joins = [(['dbo:BusCompany'], 'dbo:regionServed')]
    # bus_joinprops = {'dbo:regionServed' : ['<http://dbpedia.org/ontology/PopulatedPlace/areaTotal>',
    #                    'dbo:percentageOfAreaWater'] }
    
    # joins_from_kg(bus_pairs, bus_joinprops, bus_joins, 'busridertbl.csv', 'busriderjoin.csv')
    
    
    # soccer_pairs = [(['dbo:SoccerPlayer'], '<http://dbpedia.org/property/totalgoals>'), 
    #               (['dbo:SoccerPlayer'], '<http://dbpedia.org/property/totalcaps>')]
    # soccer_joins = [(['dbo:SoccerPlayer'], 'dbp:birthPlace')]
    # soccer_joinprops = {'dbp:birthPlace' : ['<http://dbpedia.org/ontology/PopulatedPlace/populationDensity>',
    #                    'dbp:hdi'] }
    
    # joins_from_kg(soccer_pairs, soccer_joinprops, soccer_joins, 'soccertbl.csv', 'soccerjoin.csv')
    
    # hosp_pairs = [(['dbo:Hospital'], '<http://dbpedia.org/ontology/bedCount>')]
    # hosp_joins = [(['dbo:Hospital'], '<http://dbpedia.org/property/areaServed>')]
    # hosp_joinprops = {'<http://dbpedia.org/property/areaServed>' : ['<http://dbpedia.org/ontology/PopulatedPlace/populationDensity>',
    #                    'dbp:yearPrecipitationDays'] }
    
    # joins_from_kg(hosp_pairs, hosp_joinprops, hosp_joins, 'hospitaltbl.csv', 'hospitaljoin.csv')
    
    
    # bank_pairs = [(['dbo:Bank'], '<http://dbpedia.org/property/revenue>')]
    # bank_joins = [(['dbo:Bank'], 'dbo:location')]
    # bank_joinprops = {'dbo:location' : ['<http://dbpedia.org/ontology/PopulatedPlace/populationDensity>'] }
    # joins_from_kg(bank_pairs, bank_joinprops, bank_joins, 'banktbl.csv', 'bankjoin.csv')
    
    pol_pairs = [(['dbo:Politician'], '<http://dbpedia.org/property/votes>')]
    pol_joins = [(['dbo:Politician'], 'dbo:education')]
    pol_joinprops = {'dbo:education' : ['dbo:endowment'] }
    joins_from_kg(pol_pairs, pol_joinprops, pol_joins, 'politiciantbl.csv', 'politicianjoin.csv')
    
    
    
    
    
    #TODO: we also need to implement the ability to generate the joins of these tables,
    #given relationships. Then, we can use these tables in the baseline.
    # hockey_pairs = 
    # tennis_pairs = 
    # politician_pairs = 
    # bank_pairs = 
    # soccer_pairs = 
    
    
    #construct_numfts(init_pairs)
    #create_fcm()
    
    
    
