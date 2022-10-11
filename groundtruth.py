import sqlite3

def soccer_gt():
    con = sqlite3.connect("../semanticdata/database.sqlite")
    cur = con.cursor()
    tbl_res = cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbls = [tbl[0] for tbl in tbl_res.fetchall()]
    #print(tbls)
    comp_lst = []
    
    for tbl in tbls:
        #if 'Player_Attribute'.lower() in tbl.lower():
        query = "PRAGMA foreign_key_list('" + tbl + "');"
        q_res = cur.execute(query)
        #the 2nd, 3rd, 4th entries are the 2nd table,
        #the 1st table FK name, and the 3rd table name
        #print(q_res.fetchall())
        q_mat = q_res.fetchall()
        for e in q_mat:
            fktbl = e[2]
            fkcol = e[3]
            fktblcol = e[4]
            gt1 = (tbl, fkcol)
            gt2 = (fktbl, fktblcol)
            comp_lst.append((gt1, gt2))
    
    print(comp_lst)
    
    with open('soccer_gt.txt', 'w+') as fh:
        tot_str = ''
        for c in comp_lst:
            tot_str += str(c) + '\n'
        fh.write(tot_str)

soccer_gt()