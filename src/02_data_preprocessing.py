import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def data_preprocessing(df):
    data = df.copy()
    categorical_cols = data.select_dtypes('object').columns
    numeric_cols = [col for col in data.columns if col not in categorical_cols]
    ## replace all row contains "car" to "car"
    data['purpose'] = data['purpose'].str.replace(r'.*car.*', 'car', regex=True)
    ## save data back to working folder.
    data.to_csv('.\data\data.csv', index=False)
    return df
if __name__ =='__main__':
    df = pd.read_csv('.\data\credit.csv')
    data_preprocessing(df)