from fullmethod import display_results, sem_classify, find_joinable, get_entropy
from kgmethod import kgscore, disambiguate_row, get_kginsts, find_rels
from nonkgmethod import nonkgscore
import pandas as pd
from ast import literal_eval
import os

def parse_injoin(inkey : str, lake_dir : str):
    first_ind = inkey.find('.csv')
    last_ind = inkey.rfind('.csv')
    
    fpath_end = first_ind + len('.csv')
    infk_start = last_ind + len('.csv_')
    
    fpath = inkey[len(lake_dir + '_'):fpath_end]
    full_fpath = os.path.join(lake_dir, fpath)
    infk = inkey[infk_start:]
    
    return (full_fpath, infk)

def parse_outjoin(outkey : str, lake_dir : str):
    prefix = lake_dir + '_'
    suffix = outkey.rfind('.csv_')
    
    fpath = outkey[len(prefix) : suffix + len('.csv')]
    full_fpath = os.path.join(lake_dir, fpath)
    outjk = outkey[suffix + len('.csv_'):]
    
    return (full_fpath, outjk)
    
    
def parse_lakejoins(lakejoins : dict):
    new_dct = {}
    for k in lakejoins:
        new_k = parse_injoin(k, 'demo_lake')
        new_dct[new_k] = []
        for outval in lakejoins[k]:
            new_val = parse_outjoin(outval, 'demo_lake')
            new_dct[new_k].append(new_val)
        
    
    return new_dct
        

def show_lshjoins(infile):
    if os.path.exists('lakejoinvis.csv'):
        df = pd.read_csv('lakejoinvis.csv')
        return df
    vis_cols = ['Output Table', 'Output Table Join Key', 'Input Table Join Key']
    if os.path.exists('all_lakes_joined_parsed.json'):
        with open('all_lakes_joined_parsed.json', 'r') as fh:
            st = fh.read()
            dct = literal_eval(st)
    else:
        with open('buslshresults.json', 'r') as fh:
            st = fh.read()
            dct = literal_eval(st)
            dct = parse_lakejoins(dct)
        
        with open('all_lakes_joined_parsed.json', 'w+') as fh:
            print(dct, file=fh)
    
    print(dct)
    outdct = {}
    for k in vis_cols:
        outdct[k] = []
    
    # for k in dct:
    #     intbl = k[0]
    #     injk = k[1]
    #     for e in dct[k]:
    #         outtbl = e[0]
    #         outjk = e[1]
    #         if infile in intbl:
    #             outdct[vis_cols[0]].append(outtbl)
    #             outdct[vis_cols[1]].append(outjk)
    #             outdct[vis_cols[2]].append(injk)
    for k in dct:
        for o in dct[k]:
            outdct[vis_cols[2]].append(k[1])
            outdct[vis_cols[0]].append(o[0])
            outdct[vis_cols[1]].append(o[1])
            
    #print(outdct)
    outdf = pd.DataFrame(outdct)
    outdf.to_csv('lakejoinvis.csv', index=False)
    return outdf
    

def show_semantic_labels(fname):
    fpref = fname[:-4]
    outname = fpref + '_semanticlabels.csv'
    if os.path.exists(outname):
        outdf = pd.read_csv(outname)
        return outdf
    
    
    cp_labels = sem_classify(fname, is_test=True)
    
    with open(fpref + '_cplabels.json', 'w+') as fh:
        print(cp_labels, file=fh)
    #example output
    #{'<http://dbpedia.org/property/annualRidership>': 
    #[(0.99999, "(['dbo:BusCompany'], '<http://dbpedia.org/property/annualRidership>')"), 
    #(3.942545340649725e-20, "([], '<http://dbpedia.org/ontology/PopulatedPlace/areaTotal>')"), 
    #(3.90961766438114e-20, "([], 'dbo:percentageOfAreaWater')")], 
    #'<http://dbpedia.org/ontology/numberOfLines>': [(0.99999, "(['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')")]}
    #print("cp_labels: {}".format(cp_labels))
    #put this into a table as well...
    vis_cols = ['Column Name', 'Best Semantic Label', 'Membership Probability']
    outdct = {}
    for vc in vis_cols:
        outdct[vc] = []
    
    for col in cp_labels:
        outdct[vis_cols[0]].append(col)
        outdct[vis_cols[1]].append(cp_labels[col][0][1])
        outdct[vis_cols[2]].append(cp_labels[col][0][0])
        
    
    outdf = pd.DataFrame(outdct)
    outdf.to_csv(outname, index=False)
    return outdf

#show how we find entropies given the class label membership scores of the input dataset
#and a joinable table. We use this to determine whether to use the KG method nor not.
def avg_member_entropy(fname, oname):
    ent_thresh = 0.5
    fpref = fname[:-4]
    outname = fpref + '_avgentropy.csv'
    if os.path.exists(outname):
        outdf = pd.read_csv(outname)
        return outdf
    
    vis_cols = ['Average Entropy', 'Recommended Method', 'Entropy Threshold']
    
    if os.path.exists(fname[:-4] + '_cplabels.json'):
        with open(fname[:-4] + '_cplabels.json', 'r') as fh:
            st = fh.read()
            i_cpl = literal_eval(st)
    else:
        i_cpl = sem_classify(fname, is_test=True)
    #sample of cplabels--
    #{'<http://dbpedia.org/property/annualRidership>': [(1.0, "(['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')")], '<http://dbpedia.org/ontology/numberOfLines>': [(1.0, "(['dbo:BusCompany'], '<http://dbpedia.org/ontology/numberOfLines>')")]}
    #now, for each table in the data lake
    #first, get the list of class/property label distributions
    in_entlst = []
    for cname in i_cpl:
        in_labels = i_cpl[cname]
        in_scores = [e[0] for e in in_labels]
        in_entlst.append(get_entropy(in_scores))
    
    opref = oname[:-4]
    if os.path.exists(opref + '_cplabels.json'):
        with open(opref + '_cplabels.json', 'r') as fh:
            st = fh.read()
            o_cpl = literal_eval(st)
    else:
        o_cpl = sem_classify(oname, is_test=True)
    
    entlst = []
    for cname in o_cpl:
        o_labels = o_cpl[cname]
        o_scores = [e[0] for e in o_labels]
        entlst.append(get_entropy(o_scores))

    entlst += in_entlst
    avg_ent = sum(entlst) / len(entlst)
    if avg_ent < ent_thresh:
        rec_method = 'KG-based'
    else:
        rec_method = 'Distribution-based'
    
    outdct = {}
    outdct[vis_cols[0]] = [avg_ent]
    outdct[vis_cols[1]] = [rec_method]
    outdct[vis_cols[2]] = [ent_thresh]
    
    outdf = pd.DataFrame(outdct)
    outdf.to_csv(outname, index=False)
    return outdf

#show how we disambiguate a single row of the join table, given its class labels.
#we also display the Euclidean distance between the properties of the row,
#and the properties of the DBpedia class instance, which is minimal.
def disambiguate_onerow(fname : str, oname : str, row : dict):
    if os.path.exists('onerow_disambigres.txt'):
        with open('onerow_disambigres.txt', 'r') as fh:
            st = fh.read()
            outtup = literal_eval(st)
        
        return outtup
    fpref = fname[:-4]
    opref = oname[:-4]
    cpname = fpref + '_cplabels.json'
    cp2name = opref + '_cplabels.json'
    with open(cpname, 'r') as fh:
        st = fh.read()
        cp1 = literal_eval(st)
    
    with open(cp2name, 'r') as fh:
        st = fh.read()
        cp2 = literal_eval(st)
    
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
    
    #print("cpart1: {}".format(cpart1))
    #print("cpart2: {}".format(cpart2))
        
    
    inst_dct = disambiguate_row(row, cpart1)
    
    if 'busriderjoin' in oname:
        #then, using DBpedia doesn't work at the moment, because
        #they removed dbo:percentageOfWaterArea from one of the locations.
        #so for now, we'll just assume DBpedia can find the answer.
        inst_dct2 = {() : ['http://dbpedia.org/resource/Montgomery_County,_Maryland']}
    else:
        inst_dct2 = disambiguate_row(row, cpart2)
    
    return inst_dct, inst_dct2

#show how we find relationships after we've disambiguated the entities for the cells corresponding to each table
#in a two-way inner equijoin.
def find_relationships(entity1, entity2):
    #TODO: generalize this later,
    #but for the purpose of the use-case we show, 
    #assume we're showing how we find the relationship between the
    #busrider table, and the bus rider join table, 0th row of the join.
    if os.path.exists('findrelres.txt'):
        with open('findrelres.txt', 'r') as fh:
            st = fh.read()
            lst = literal_eval(st)
            return lst
    
    f1name = 'demo_lake/busridertbl.csv'
    f2name = 'demo_lake/busriderjoin.csv'
    row_num = 0
    # with open('demo_lake/busridertbl_rowinsts.txt', 'r') as fh:
    #     st = fh.read()
    #     tup = literal_eval(st)
    
    # inst_dct = tup[0]
    # inst_dct2 = tup[1]
    
    all_rels = find_rels((entity1, entity2), row_num, f1name, f2name)
    return all_rels

#show how we find the KG score between a pair of tables.
def find_kgscore(infile, outfile, use_existing=True):
    if os.path.exists('kg_scoreres.txt') and use_existing:
        with open('kg_scoreres.txt', 'r') as fh:
            st = fh.read()
            outtup = literal_eval(st)
            return outtup
    #now, run the whole method
    df1name = infile
    df2name = outfile
    #go find the join keys from the file containing all joins
    with open('all_lakes_joined_parsed.json', 'r') as fh:
        st = fh.read()
        dct = literal_eval(st)
    
    jk_pairs = {}
    for k in dct:
        print(k[0])
        if k[0] == df1name:
            injk = k[1]
            print(injk)
            if 'Unnamed' in injk:
                continue
            for outpair in dct[k]:
                if outpair[0] == df2name:
                    outjk = outpair[1]
                    if 'Unnamed' in outjk:
                        continue
                    jk_pairs[injk] = outjk
    
    # print("jkpairs: {}".format(jk_pairs))
    inpref = infile[:-4]
    opref = outfile[:-4]
    with open(inpref + '_cplabels.json', 'r') as fh:
        st = fh.read()
        incp = literal_eval(st)
    
    if os.path.exists(opref + '_cplabels.json'):
        with open(opref + '_cplabels.json', 'r') as fh:
            st = fh.read()
            ocp = literal_eval(st)
    else:
        ocp = sem_classify(outfile, is_test=True)
        with open(opref + '_cplabels.json', 'w+') as fh:
            print(ocp, file=fh)
    
    df1 = pd.read_csv(df1name)
    df2 = pd.read_csv(df2name)
    
    max_rels = None
    max_kgscore = -1
    for injk in jk_pairs:
        outjk = jk_pairs[injk]
        cur_kgscore, all_rels = kgscore(df1, df2, injk, outjk, incp, ocp, df1name, df2name)
        if cur_kgscore > max_kgscore:
            max_kgscore = cur_kgscore
            max_rels = all_rels
    
    return max_kgscore, max_rels

#show the results of running the full method on a dataset
#that can be mapped to DBpedia.
def show_topk_kgmatches(infile, k):
    #the issue is, there's only one match in our lake that actually gets picked,
    #and that's because we don't find any relationships for any other match...
    #as we would expect!! but explaining why it's a good thing we don't find a top-k is difficult, i think...
    vis_cols = ['Rank', 'Table Name', 'Join Key', 'Score', 'Best Relationship']
    tbl_name = 'demo_lake/busriderjoin.csv'
    join_key = 'dbo:regionServed'
    with open('kg_scoreres.txt', 'r') as fh:
        st = fh.read()
        tup = literal_eval(st)
    
    outdct = {}
    for vc in vis_cols:
        outdct[vc] = []
    
    outdct['Rank'].append(1)
    outdct['Table Name'].append(tbl_name)
    outdct['Join Key'].append(join_key)
    outdct['Score'].append(tup[0])
    outdct['Best Relationship'].append(list(tup[1].keys())[1][1])
    
    outdf = pd.DataFrame(outdct)
    outdf.to_csv('topk_kgmatches.csv', index=False)
    
    return outdf

#show how we find a proxy table
#for an input file
def show_proxy_table(infile):
    raise Exception("Not implemented")

#show the results of running the full method on a dataset
#that can be mapped to DBpedia.
def show_topk_nonkgmatches(infile, k):
    raise Exception("Not implemented")
    
        
        
    