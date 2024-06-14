import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import os


def load_dataset(categ=False, scaler=None):
    """
    Load the dataset and preprocess it
    :param categ: boolean, if True, the binary columns are converted to categorical
    :param scaler: string,
        if 'robust', the data is scaled using RobustScaler
        if 'standard', the data is scaled using StandardScaler
        if 'center', the mean is subtracted from the data
    :return: pandas dataframe
    """
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "Unsupervised Learning 23-24 - Project Dataset.csv",
    )

    df_raw = pd.read_csv(dataset_path, sep=";").iloc[:, 1:-2]

    object_columns = df_raw.select_dtypes(include=["object"]).columns
    for column in object_columns:
        try:
            df_raw[column] = pd.to_numeric(
                df_raw[column].str.replace(",", "."), errors="coerce"
            )
        except:
            print(f"Error for column: {column}, check the csv file.")

    # options for preprocessing continuous variables
    if scaler in ["robust", "standard"]:
        sc = None
        if scaler == "standard":
            sc = StandardScaler()
        elif scaler == "robust":
            sc = RobustScaler()

        df = df_raw.copy(deep=True)
        cont = list(filter(lambda col: col[-2:] != "=0", df_raw.columns))
        df[cont] = sc.fit_transform(df_raw[cont])
        df_raw = df

    if scaler == "center":
        df_raw = df_raw - df_raw.mean()

    # options for preprocessing categorical variables
    if categ:
        for column in df_raw:
            if column[-2:] == "=0":
                df_raw[column] = df_raw[column].astype("category")

    return df_raw
