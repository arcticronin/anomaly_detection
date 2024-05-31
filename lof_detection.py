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
    # initialize the model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=0.05,
                             metric="precomputed")
    # fit the model
    # lof.fit(df)
    # compute the LOF scores
    # lof_scores = lof.negative_outlier_factor_ ## are scores or distances
    predictions = lof.fit_predict( gower.gower_matrix(df) )
    return  np.array(((predictions - 1) / 2), dtype=int)

if __name__ == '__main__':
    main()