import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import importlib

# custom libraries
import utils

importlib.reload(utils)
import preprocessing

importlib.reload(preprocessing)


def main(distance_matrix):
    """main

    Args:
        distance_matrix (numpy array): square n x n matrix with the pairwise distances between the data points

    Returns:
        numpy array : contains the labels of the outliers, -1 if the point is an outlier, 0 otherwise
    """
    df = preprocessing.load_dataset()
    # fit the model to the distance matrix
    # we use the gower distance as the metric
    # hyperparameter selected in the dbscan notebook
    db = DBSCAN(eps=0.045, min_samples=14, metric="precomputed")
    labels = db.fit_predict(distance_matrix)
    outliers_labels = np.array(-1 * (labels == -1), dtype=int)
    # outliers will be labeled as -1, inliers as 0
    return outliers_labels


if __name__ == "__main__":
    main()
