import pandas as pd
from sklearn.preprocessing import RobustScaler
import os


def load_dataset(categ = False, scaler = False):
    '''
    Load the dataset and preprocess it
    :param categ: boolean, if True, the columns that end with '=0' are converted to categorical
    :param scaler: string,
        if 'robust', the data is scaled using RobustScaler
        if 'center', the mean is subtracted from the data
    :return: pandas dataframe
    '''
    dataset_path = os.path.join(os.path.dirname(__file__),
                                '..',
                                'data',
                                'Unsupervised Learning 23-24 - Project Dataset.csv')

    df_raw = pd.read_csv(dataset_path, sep=';').iloc[:, 1:-2]
    
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
        df_raw = df

    if scaler == 'center':
        df_raw -= df_raw.mean()
        
    if categ == True:
        for column in df_raw:
            if column[-2:] == '=0':
                #print(column)
                df_raw[column] = df_raw[column].astype('category')

    return df_raw

