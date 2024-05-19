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

def main():
    # load the dataset
    df = preprocessing.load_dataset()
    # set the number of neighbors
    n_neighbors = 15
    # initialize the model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
    # fit the model
    lof.fit(df)
    # compute the LOF scores
    lof_scores = lof.negative_outlier_factor_ ## are scores or distances

    ## TODO get a better threshold
    ## use knee method
    #import kneed
    #from kneed import KneeLocator
    #kn = KneeLocator(range(1, len(lof_scores)), np.sort(lof_scores), curve='convex', direction='decreasing')


    predictions = lof.fit_predict(df)
    outlier_indices = np.array([0 if i ==False else -1 for i in -1 * (lof_scores < -6)])
    return  outlier_indices

if __name__ == '__main__':
    main()