

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


#given a 
def kgscore(df1, df2, jk1, jk2, cp1 : dict, cp2 : dict):
    joindf = df1.merge(left_on=jk1, right_on=jk2)
    found_rels = {}
    #first, find the groups of class/property pairs with the same class,
    #for each inputted table.
    cprops1 = []
    cprops2 = []
    for cname in cp1:
        memvec = cp1[cname]
        
        #best_cl = 
        
    #find relationships in each row

