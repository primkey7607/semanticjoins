import os
import pickle
import pandas as pd

"""
Purpose: Given an input dataset and a data lake, run the full pipeline.
"""

def sem_classify(fname):
    #first, load models into memory...
    fcm_models = {}
    for f in os.listdir():
        if f.startswith('fcm_') and f.endswith('model.pkl'):
            mtp = f[4:-9]
            with open(f, 'r') as fh:
                model = pickle.load(fh)
            
            fcm_models[mtp] = model
    
    #then, load the table as a list of columns
    df = pd.read_csv(fname)
    
    for c in df.columns:
        col_lst = df[c].to_list()
        

