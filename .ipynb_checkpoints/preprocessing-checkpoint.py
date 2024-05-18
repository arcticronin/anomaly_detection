import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def load_dataset(categ = False, scaler = False):
    df_raw = pd.read_csv('Unsupervised Learning 23-24 - Project Dataset.csv', sep=';').iloc[:,1:-2]
    
    object_columns = df_raw.select_dtypes(include=['object']).columns
    for column in object_columns:
        try:
            df_raw[column] = pd.to_numeric(df_raw[column].str.replace(',', '.'), errors='coerce')
        except ValueError:
            print(f"Conversion failed for column: {column}. It may contain non-numeric values.")
    
    if scaler == 'robust':  
        scaler = RobustScaler()
        df = df_raw.copy(deep=True)
        cont = list(filter(lambda col: col[-2:] != '=0', df_raw.columns))
        df[cont] = scaler.fit_transform(df_raw[cont])
        
    if categ == True:
        for column in df_raw:
            if column[-2:] == '=0':
                #print(column)
                df_raw[column] = df_raw[column].astype('category')
    return df_raw

