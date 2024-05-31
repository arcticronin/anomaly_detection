import pandas as pd
import numpy as np
#distance
import gower
#
import importlib
import utils
importlib.reload(utils) #debug - remove
import preprocessing
importlib.reload(preprocessing) #debug - remove
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def main():
    # load the dataset
    df = preprocessing.load_dataset()
    # set the number of neighbors
    distance_matrix = gower.gower_matrix(df)

    ## selected k = 14
    # Initialize the NearestNeighbors model with precomputed distances
    k = 16
    neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')

    # Fit the model to the distance matrix
    neigh.fit(distance_matrix)

    # Compute the distances and indices of the k-th nearest neighbors
    distances, indices = neigh.kneighbors(distance_matrix)

    # Extract the distances to the k-th nearest neighbor
    distances_kth = distances[:, -1]

    # Sort the distances to the k-th nearest neighbor
    distances_kth_sorted = np.sort(distances_kth)

    # Find the knee point
    kl = KneeLocator(np.arange(len(distances_kth_sorted)),
                     distances_kth_sorted,
                     S=3,
                     curve='convex',
                     direction='increasing')

    print(f"knee = {kl.knee_y}")
    # Label outliers: points with a distance greater than the knee point distance are considered outliers
    outlier_labels = (distances_kth > kl.knee_y) * -1

    # Trovare gli indici degli outliers nel dataset
    return outlier_labels

if __name__ == '__main__':
    main()