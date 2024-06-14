import pandas as pd
import numpy as np

# distance
import gower

#
import importlib
import utils

importlib.reload(utils)  # debug - remove
import preprocessing

importlib.reload(preprocessing)  # debug - remove
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def main(distance_matrix):
    # selected k = 16 from hyperparameter tuning in knn notebook
    k = 16
    neigh = NearestNeighbors(n_neighbors=k, metric="precomputed")

    # fit the model to the distance matrix
    neigh.fit(distance_matrix)

    # compute and sort the distances and indices of the 16th nearest neighbors
    distances, indices = neigh.kneighbors(distance_matrix)
    distances_kth = distances[:, -1]
    distances_kth_sorted = np.sort(distances_kth)

    # use a knee based approach to find the threshold
    kl = KneeLocator(
        np.arange(len(distances_kth_sorted)),
        distances_kth_sorted,
        S=3,
        curve="convex",
        direction="increasing",
    )
    print(f"knee = {kl.knee_y}")

    # points with a distance greater than the knee point distance are flagged as outliers
    outlier_labels = (distances_kth > kl.knee_y) * -1
    return outlier_labels


if __name__ == "__main__":
    main()
