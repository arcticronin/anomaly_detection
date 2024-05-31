import pandas as pd
import numpy as np

#distance
import gower

import importlib
import utils
importlib.reload(utils) #debug - remove
import preprocessing
importlib.reload(preprocessing) #debug - remove

from sklearn.cluster import DBSCAN


## TODO fix
def main():
    df = preprocessing.load_dataset()
    # fit the model to the distance matrix
    # we use the gower distance as the metric
    # hyperparameter selected in the dbscan notebook
    db = DBSCAN(eps=0.045,
                min_samples=14,
                metric='precomputed')
    labels = db.fit_predict(gower.gower_matrix(df))
    outliers_labels = np.array(-1 * (labels == -1), dtype=int)
    # outliers will be labeled as -1, inliers as 0
    return outliers_labels

if __name__ == '__main__':
    main()