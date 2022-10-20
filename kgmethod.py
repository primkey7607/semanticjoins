from ast import literal_eval
from SPARQLWrapper import SPARQLWrapper, JSON
import copy

"""
Purpose: Index the KG method query results. The actual method has to issue SPARQL queries,
which can be quite expensive unless you dump the data onto disk.
So in this file, we index the KG method results. We use them when
we run the full method.

 Design: Given the input table and its semantic labels 
(we can assume these are correct, because the user would probably know what they are)
For each table in the data lake, (1) semantically label its columns. (2) join that table with the input

(3) Then, for each tuple in the join output,
disambiguate each group of columns represented by class/property pairs with the same class
to a KG instance. The output will be a set of class instances for the tuple. Some of these instances will belong to the input table tuple values,
and others will belong to the lake table side.
(4) Check whether at least one of the instances and the other have a path less than length 2 between them. If so, store the relationship.
(5) After having done this, we should have a list of relationships per tuple. 
Score the join as the proportion of tuples in which the maximum occurring most frequent relationship is observed.

So, the input should be two tables, along with column-semantic label pairs,
and the output should be the proportion of related values, as described above.
And that's all this file should contain. The full method will contain the rest,
and use helper functions from here.
The reason why is--we're deciding whether to use the KG method, or non-KG method
for each pair of tables.
"""

#relationships that don't actually indicate anything...
g_rels = ['<http://dbpedia.org/property/wikiPageUsesTemplate>',
    '<http://dbpedia.org/ontology/wikiPageRevisionID>',
    '<http://dbpedia.org/ontology/wikiPageWikiLink>',
    '<http://dbpedia.org/ontology/wikiPageLength>',
    '<http://dbpedia.org/ontology/wikiPageID>',
    '<http://dbpedia.org/property/n>',
    '<http://dbpedia.org/ontology/abstract>',
    '<http://purl.org/dc/terms/subject>',
    '<http://www.w3.org/ns/prov#wasDerivedFrom>',
    '<http://dbpedia.org/property/isoCodeType>',
    '<http://dbpedia.org/property/colwidth>',
    '<http://dbpedia.org/property/e>',
    '<http://dbpedia.org/ontology/thumbnail>',
    '<http://xmlns.com/foaf/0.1/depiction>',
    '<http://purl.org/linguistics/gold/hypernym>',
    '<http://xmlns.com/foaf/0.1/isPrimaryTopicOf>',
    '<http://dbpedia.org/property/txt>',
    '<http://dbpedia.org/property/imageSize>']

def disambiguate_row(row, cpt):
    cl_insts = {}
    
    for cl1 in cpt:
        q1 = 'select ?subject where { '
        for cl in cl1:
            q1 += '?subject a ' + cl + ' . '
        
        cl1props = cpt[cl1]
        flt_use = False
        q1_filter = ''
        for i,p in enumerate(cl1props):
            q1 += '?subject ' + p[1] + ' ?val' + str(i) + ' . '
            if flt_use:
                q1_filter += '&& '
            else:
                q1_filter = 'FILTER ( '
                flt_use = True
            q1_diff = ' ( xsd:float( ?val' + str(i) + ') - ' + str(row[p[0]]) + ')'
            q1_filter += q1_diff + '*' + q1_diff + ' < 0.5 '
        
        if flt_use:
            q1 += q1_filter + ') '
        #TODO: we put a limit for efficiency, but this could easily make it so the
        #answer we want doesn't appear here, so we may have to get rid of the limit
        q1 += ' } LIMIT 10'
        print("Querying Endpoint q1")
        sparql = SPARQLWrapper("https://dbpedia.org/sparql/", agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
        sparql.setQuery(q1)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        q1subs = []
        for b in results['results']['bindings']:
            q1subs.append(b['subject']['value'])
        cl_insts[cl1] = q1subs
    
    return cl_insts
        
    

#given a dictionary of row values and dictionaries of properties corresponding
#to the same class, disambiguate to the KG and return candidate instances
def get_kginsts(row, cpt1 : dict, cpt2 : dict):
    #guess instances from row values
    cl1_insts = disambiguate_row(row, cpt1)
    cl2_insts = disambiguate_row(row, cpt2)
    
    return cl1_insts, cl2_insts

def find_rels(kg_insts):
    #for each pair of instances here, find relationships
    #such that there are no garbage relationships
    cl1_insts = kg_insts[0]
    cl2_insts = kg_insts[1]
    all_rels = set()
    
    for cl1 in cl1_insts:
        for cl2 in cl2_insts:
            #set a threshold first--if we end up with too many empty answers,
            #then it's just true that instances from these classes are generally not related,
            #so just stop.
            #c1len = len(cl1_insts[cl1])
            #c2len = len(cl2_insts[cl2])
            thresh = 5
            insts1 = cl1_insts[cl1]
            insts2 = cl2_insts[cl2]
            
            empty_cnt = 0
            for inst1 in insts1:
                for inst2 in insts2:
                    if empty_cnt >= thresh:
                        continue
                    grel_st = ', '.join(g_rels)
                    query = 'select ?rel where { ' + inst1 + ' ?rel ' + inst2
                    query += ' FILTER ( ?rel NOT IN ( ' + grel_st + ' ) ) }'
                    opquery = 'select ?rel where { ' + inst2 + ' ?rel ' + inst1
                    opquery += ' FILTER ( ?rel NOT IN ( ' + grel_st + ' ) ) }'
                    sparql = SPARQLWrapper("https://dbpedia.org/sparql/", agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()
                    
                    sparql.setQuery(opquery)
                    sparql.setReturnFormat(JSON)
                    opresults = sparql.query().convert()
                    
                    if results['results']['bindings'] == [] and opresults['results']['bindings'] == []:
                        empty_cnt += 1
                    else:
                        for b in results['results']['bindings']:
                            new_rel = (cl1, b['rel']['value'], cl2)
                            all_rels.add(new_rel)
                        
                        for b in opresults['results']['bindings']:
                            new_rel = (cl2, b['rel']['value'], cl1)
                            all_rels.add(new_rel)
    
    return list(all_rels)

def verify_rels(all_rels, kg_insts, existing_rels):
    new_all = set()
    cl1_insts = kg_insts[0]
    cl2_insts = kg_insts[1]
    sparql = SPARQLWrapper("https://dbpedia.org/sparql/", agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
    
    for cl1 in cl1_insts:
        for cl2 in cl2_insts:
            for ar in all_rels:
                if ar in existing_rels:
                    continue
                if ar[0] == cl1 and ar[2] == cl2:
                    for inst1 in cl1_insts[cl1]:
                        for inst2 in cl2_insts[cl2]:
                            query = 'ask { ' + inst1 + ' ' + ar[1] + ' ' + inst2 + ' . } '
                            sparql.setQuery(query)
                            sparql.setReturnFormat(JSON)
                            results = sparql.query().convert()
                            rel_exists = results['boolean']
                            if rel_exists:
                                new_all.add(ar)
                        
                
                if ar[0] == cl2 and ar[2] == cl1:
                    for inst1 in cl1_insts[cl1]:
                        for inst2 in cl2_insts[cl2]:
                            query = 'ask { ' + inst2 + ' ' + ar[1] + ' ' + inst1 + ' . } '
                            sparql.setQuery(query)
                            sparql.setReturnFormat(JSON)
                            results = sparql.query().convert()
                            rel_exists = results['boolean']
                            if rel_exists:
                                new_all.add(ar)
    
    return new_all
                            
                                
                            
                    
    

#given two datasets, two join keys, and two dictionaries for each dataset, where
#each dictionary has column names as keys and the values are lists of tuples where the first is the membership score,
#and the second is the class/property pair,
#return the proportion of tuples in the join that we find are related through the KG.
def kgscore(df1, df2, jk1, jk2, cp1 : dict, cp2 : dict):
    joindf = df1.merge(left_on=jk1, right_on=jk2)
    found_rels = {}
    #first, find the groups of class/property pairs with the same class,
    #for each inputted table
    cprops1 = []
    cpart1 = {}
    cprops2 = []
    cpart2 = {}
    for cname in cp1:
        #the most likely class
        ml_cl = max(cp1[cname], key=lambda x: x[0])[1]
        cprops1.append((cname, ml_cl))
    
    #now, partition cp1 by class
    for cprop in cprops1:
        ctup = literal_eval(cprop[1])
        cn = cprop[0]
        c_cl = tuple(ctup[0])
        if c_cl in cpart1:
            cpart1[c_cl].append((cn, ctup[1]))
        else:
            cpart1[c_cl] = [(cn, ctup[1])]
    
    for cname in cp2:
        #the most likely class
        ml_cl = max(cp2[cname], key=lambda x: x[0])[1]
        cprops2.append((cname, ml_cl))
    
    #now, partition cp2 by class
    for cprop in cprops2:
        cn = cprop[0]
        ctup = literal_eval(cprop[1])
        c_cl = tuple(ctup[0])
        if c_cl in cpart2:
            cpart2[c_cl].append((cn, ctup[1]))
        else:
            cpart2[c_cl] = [(cn, ctup[1])]
    
    all_rels = {}
    #now, for each row in the join result, disambiguate the values to instances
    #and check if there's a path between KG instances
    for j,r in enumerate(joindf.to_dict(orient='records')):
        kg_insts = get_kginsts(r, cpart1, cpart2)
        #find the new relationships among these found instances
        kg_rels = find_rels(kg_insts)
        existing_rels = []
        for rel in kg_rels:
            if rel in all_rels:
                all_rels[rel] += 1
                existing_rels.append(rel)
            else:
                all_rels[rel] = 1
            
        #but most likely, we won't be as lucky as the above. We'll have to 
        #verify the relationships we already have through sparql
        #while skipping those we found before, and also just discovered.
        verified_rels = verify_rels(all_rels, kg_insts, existing_rels)
        for v in verified_rels:
            all_rels[v] += 1
    
    #now, the score is the maximum proportion of rows that any one relationship covers
    max_score = max([all_rels[ar] for ar in all_rels]) / joindf.shape[0]
    return max_score, all_rels
    
        
        

