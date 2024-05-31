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
    df = preprocessing.load_dataset()
    isolation_forest = IsolationForest(contamination=0.05, bootstrap=True,
                                       random_state=123)  # contamination è la percentuale di outliers che dico di trovare

    outlier_labels = isolation_forest.fit_predict(df)

    return outlier_labels

if __name__ == '__main__':
    main()