import datasketch
import os
from find_lsh import colsets_of, basic_extract

"""
Purpose: when i ran lsh using the below functions,
I got reliable matches. So, something's wrong with the code I wrote in find_lsh.py,
and I don't think it's worth figuring out what.
"""

def build_lsh(lake_dir):
    new_lsh = datasketch.MinHashLSH(threshold=0.5, num_perm=128)
    for f in os.listdir(lake_dir):
        fullf = os.path.join(lake_dir, f)
        if fullf.endswith('.csv'):
            outdf = basic_extract(fullf)
            outcolsets = colsets_of(outdf, fullf)
            for i,colset in enumerate(outcolsets[0]):
                newmh = datasketch.MinHash(num_perm=128)
                for d in colset:
                    newmh.update(str(d).encode('utf8'))
                    new_name = outcolsets[1][i]
                print("name: {}".format(new_name))
                print("sample value: {}".format(next(iter(colset))))
                new_lsh.insert(new_name, newmh)
    
    return new_lsh

def query_lsh(infile, lsh_ind):
    res_out = {}
    new_lsh = lsh_ind
    indf = basic_extract(infile)
    incolsets = colsets_of(indf, infile)
    for i,colset in enumerate(incolsets[0]):
        newmh = datasketch.MinHash(num_perm=128)
        for d in colset:
                newmh.update(str(d).encode('utf8'))
        new_name = incolsets[1][i]
        print("name: {}".format(new_name))
        print("sample value: {}".format(next(iter(colset))))
        matches = new_lsh.query(newmh)
        print("matches: {}".format(matches))
        res_out[new_name] = matches
    
    return res_out

def test_lsh(infile, lake_dir):
    res_out = {}
    #first, build the index
    new_lsh = datasketch.MinHashLSH(threshold=0.5, num_perm=128)
    for f in os.listdir(lake_dir):
        fullf = os.path.join(lake_dir, f)
        if fullf.endswith('.csv'):
            outdf = basic_extract(fullf)
            outcolsets = colsets_of(outdf, fullf)
            for i,colset in enumerate(outcolsets[0]):
                newmh = datasketch.MinHash(num_perm=128)
                for d in colset:
                    newmh.update(str(d).encode('utf8'))
                    new_name = outcolsets[1][i]
                print("name: {}".format(new_name))
                print("sample value: {}".format(next(iter(colset))))
                new_lsh.insert(new_name, newmh)
    
    #now, let's query the built index.
    indf = basic_extract(infile)
    incolsets = colsets_of(indf, infile)
    for i,colset in enumerate(incolsets[0]):
        newmh = datasketch.MinHash(num_perm=128)
        for d in colset:
                newmh.update(str(d).encode('utf8'))
        new_name = incolsets[1][i]
        print("name: {}".format(new_name))
        print("sample value: {}".format(next(iter(colset))))
        matches = new_lsh.query(newmh)
        print("matches: {}".format(matches))
        res_out[new_name] = matches
    
    return res_out

if __name__ == "__main__":
    # lake_lsh = build_lsh('demo_lake')
    # busridermatches = query_lsh('demo_lake/busridertbl.csv', lake_lsh)
    # with open('buslshresults.json', 'w+') as fh:
    #     print(busridermatches, file=fh)
    # busridermatches = test_lsh('demo_lake/busridertbl.csv', 'demo_lake')
    # with open('buslshresults.json', 'w+') as fh:
    #     print(busridermatches, file=fh)
    
    etfmatches = test_lsh('demo_lake/ETF prices.csv', 'demo_lake')
    with open('etflshresults.json', 'w+') as fh:
        print(etfmatches, file=fh)
        