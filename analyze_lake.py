import pandas as pd
import os

def check_na():
    for f in os.listdir('demo_lake'):
        if f.endswith('.csv'):
            fullf = os.path.join('demo_lake', f)
            df = pd.read_csv(fullf)
            na_dct = df.isna().any()
            for c in na_dct.index:
                #print(c)
                if na_dct[c]:
                    print(fullf + ' has NA values at column: {}'.format(c))

def fix_na():
    for f in os.listdir('demo_lake'):
        if f.endswith('.csv'):
            fullf = os.path.join('demo_lake', f)
            df = pd.read_csv(fullf)
            if df.isnull().values.any():
                df = df[~df.isnull().any(axis=1)]
                if df.shape[0] < 200:
                    print("CAUTION: Significantly Downsized Table: {} to size {}".format(fullf, df.shape[0]))
                df.to_csv(fullf)
            

if __name__ == "__main__":
    #check_na()
    fix_na()

