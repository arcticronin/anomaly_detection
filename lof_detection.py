import pandas as pd
import numpy as np
import gower
import importlib
from sklearn.neighbors import LocalOutlierFactor

## custom libraries
import utils
import preprocessing
importlib.reload(preprocessing) #debug - remove
importlib.reload(utils) #debug - remove

def main():
    df = preprocessing.load_dataset()
    n_neighbors = 16
    # 16 has been selected from hyperparameter tuning of knn
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=0.05,
                             metric="precomputed")
    # fit the model and get the predictions
    predictions = lof.fit_predict(gower.gower_matrix(df))
    # prediction will contain 1 for inliers and -1 for outliers
    # we convert it in 0 for inliers and -1 for outliers
    return  np.array(((predictions - 1) / 2), dtype=int)

if __name__ == '__main__':
    main()