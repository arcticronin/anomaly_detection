import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#distance
import gower

#
import importlib
import utils
importlib.reload(utils) #debug - remove
import preprocessing
importlib.reload(preprocessing) #debug - remove

# preprocessing tools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


## TODO fix
def main():
    # load the dataset
    df = preprocessing.load_dataset()
    # set the number of neighbors
    dist_matrix = gower.gower_matrix(df)
    isolation_forest = IsolationForest(contamination=0.1, bootstrap=True,
                                       random_state=123)  # contamination è la percentuale di outliers che dico di trovare
    isolation_forest.fit(dist_matrix)

    # Identificare gli outliers nel datasetgit v
    outlier_labels = isolation_forest.predict(dist_matrix)

    # Trovare gli indici degli outliers nel dataset
    return outlier_labels

if __name__ == '__main__':
    main()