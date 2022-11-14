from datasketch import MinHash, MinHashLSH
import pandas as pd
import pickle
import os
from ast import literal_eval

"""
Purpose: generate the overlap similarity-based joins
by building a Minhash LSH index and querying the index.
The output will be a dictionary whose keys are the columns of the input file,
and values are lists of data lake columns that are similar to the input column.
"""

#given the file name, fname, of a csv file,
#return the table as a pandas dataframe
def basic_extract(fname):
    tbl = pd.read_csv(fname)
    return tbl

#given tbl as a pandas dataframe, and the directory to which it belongs, tbl_dir
#return the sets of values in each of its columns
def colsets_of(tbl, tbl_dir):
    res_cols = []
    cnames = []
    tbl_st = tbl_dir.replace(os.sep, '_')
    #TODO: add sampling if this is running too slow
    for c in tbl.columns:
        clst = tbl[c].to_list()
        cdt = tbl.dtypes[c]
        #NOTE: we cannot just convert this list to a set here.
        #we need to account for special characters, etc.
        c_clean = []
        for cval in clst:
            if cval == None and cdt == 'int':
                c_clean.append(float("nan"))
            elif cval == None and cdt == 'object':
                c_clean.append('')
            elif cval == None and cdt == 'float':
                c_clean.append(float("nan"))
            else:
                c_clean.append(cval)
        
        c_set = set(c_clean)
        res_cols.append(c_set)
        cnames.append(tbl_st + '_' + c)
    
    return res_cols, cnames

#given a list of file names in the same folder,
#generate and pickle the minhashes, and store them in a specified repo
def store_mhs(flst, pkl_loc):
    col2mh = {}
    for f in flst:
        tbl_name = f.replace(os.sep, '_')
        tbl = basic_extract(f)
        fcols, fcnames = colsets_of(tbl, f)
        mhs = [MinHash(num_perm=128)] * len(fcols)
    
        for i,s in enumerate(fcols):
            for d in s:
                mhs[i].update(str(d).encode('utf8'))
    
        for i,mh in enumerate(mhs):
            mh_name = tbl_name + '_col' + str(i) + '_mh'
            with open(os.path.join(pkl_loc, mh_name + '.pkl'), 'wb') as fh:
                pickle.dump(mh, fh)
            col2mh[fcnames[i]] = mh_name
    
    #store the mapping of column to minhash as well
    with open('mh_dict.json', 'w+') as fh:
        print(col2mh, file=fh)
    

#given a directory of pickled minhash objects,
#construct and store a Minhash LSH index
def build_lsh(pkl_loc, outname, thresh=0.5):
    lsh = MinHashLSH(threshold=thresh, num_perm=128)
    
    for f in os.listdir(pkl_loc):
        if not f.endswith('.pkl'):
            continue
        fullpath = os.path.join(pkl_loc, f)
        with open(fullpath, 'rb') as fh:
            mh = pickle.load(fh)
        fname = f[:-4]
        lsh.insert(fname, mh)
    
    with open(outname, 'wb') as fh:
        pickle.dump(lsh, fh)

#get all minhashes
def gen_mh_from(colset, colname, fname):
    new_mh = MinHash(num_perm=128)
    mh_name = fname + '_' + colname
    for d in colset:
        new_mh.update(str(d).encode('utf8'))
    
    return { mh_name : new_mh }

def get_fcnames(hashes, mhdctname):
    fclst = []
    for h in hashes:
        fcname_cand = [k for k in mhdctname if mhdctname[k] == h]
        fcname = fcname_cand[0]
        fclst.append(fcname)
    
    return fclst

def construct_gt(infname, outname):
    all_joins = {}
    inheader = pd.read_csv(infname, nrows=0).columns.to_list()
    for f in os.listdir('demo_lake'):
        fullf = os.path.join('demo_lake', f)
        if fullf.endswith('.csv') and fullf != infname:
            outheader = pd.read_csv(fullf, nrows=0).columns.to_list()
            for c in outheader:
                if c in inheader and 'Unnamed' not in c:
                    if (infname, c) in all_joins:
                        all_joins[(infname, c)].append((fullf, c))
                    else:
                        all_joins[(infname, c)] = [(fullf, c)]
    
    with open(outname, 'w+') as fh:
        print(all_joins, file=fh)
                    
            
            
            
    
#given a filename and a lsh index, return a list of all the joinable tables
#to the input file
def query_lsh(infname, mhdctname, lsh_ind, outname, is_gt=True):
    if is_gt:
        construct_gt(infname, outname)
    else:
        all_joins = {}
        intbl = basic_extract(infname)
        with open(mhdctname, 'r') as fh:
            st = fh.read()
            mhdct = literal_eval(st)
        
        with open(lsh_ind, 'rb') as fh:
            lsh = pickle.load(fh)
        
        incols, innames = colsets_of(intbl, infname)
        
        #now, query the index
        for i,inc in enumerate(incols):
            cmhent = gen_mh_from(inc, innames[i], infname)
            cmh_name = list(cmhent.keys())[0]
            cmh = cmhent[cmh_name]
            hashes = lsh.query(cmh)
            #now, we need to reconstruct the file-column names for these minhashes
            fc_lst = get_fcnames(hashes, mhdct)
            all_joins[cmh_name] = fc_lst
        
        with open(outname, 'w+') as fh:
            print(all_joins, file=fh)
        

if __name__ == "__main__":
    #do a simple test, to start
    flst = [os.path.join('demo_lake', f) for f in os.listdir('demo_lake')]
    inpfile = 'demo_lake/busridertbl.csv'
    
    store_mhs(flst, 'lake_mhs')
    build_lsh('lake_mhs', 'lake_lshind.pkl', thresh=0.4)
    
    query_lsh(inpfile, 'mh_dict.json', 'lake_lshind.pkl', 'all_lake_joins.json', is_gt=True)
    

