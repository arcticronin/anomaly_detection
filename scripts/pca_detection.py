import preprocessing
import utils
import gower
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import chi2


def main():
    df = preprocessing.load_dataset()
    dist_matrix = gower.gower_matrix(df)

    # Design parameters (more details in the pca notebook)
    # minimum number of compontent that explain 0.98 of the variance
    NCOMP = 8

    pca = PCA(n_components=NCOMP)
    # fit and transform the data
    pca_result = pca.fit_transform(dist_matrix)
    print('PCA: explained variation per principal component: {}'
    .format(pca.explained_variance_ratio_.round(5)))

    # set the alpha value to indicate the percentil of the chi-squared distribution
    alpha = 0.99
    # compute chi-squared for given alpha and degrees of freedom
    chi_2 = chi2.ppf(alpha, df=NCOMP)

    # eigenvalues of the covariance matrix
    lambdas = np.sqrt(pca.explained_variance_)

    ## we take the sum of the squared coordinates divided by the eigenvalues
    ## if its greater than the chi2 value we consider it an outlier
    outlier_indices = -(1 * (np.sum((pca_result ** 2) / np.transpose(lambdas), axis=1) > chi_2))
    return outlier_indices

if __name__ == '__main__':
    main()