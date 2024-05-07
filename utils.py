from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch

## globals
alpha = 2/10
binary_indices = [i for i in range(2, 16)]
continuous_indices = [0] + [i for i in range(16, 20)]

def dist(x, y): ## x :numpy array y: numpy array -> d : float
    cd = np.mean(np.abs(x.iloc[continuous_indices] - y.iloc[continuous_indices]))
    bd = np.sum(x.iloc[binary_indices] != y.iloc[binary_indices])
    d = alpha * cd + (1 - alpha) * bd  
    return d


def gower_loss(x_original, x_reconstructed, binary_indices, continuous_indices):
    # Binary indices handling
    alpha = len(binary_indices)/(len(binary_indices) + len(continuous_indices))
    x_bin_original = x_original[:, binary_indices]
    x_bin_reconstructed = torch.sigmoid(x_reconstructed[:, binary_indices])  # Use sigmoid to squash outputs

    # Binary loss (using BCE)
    binary_loss = F.binary_cross_entropy(x_bin_reconstructed, x_bin_original, reduction='mean')

    # Continuous data handling
    continuous_indices = [i for i in range(x_original.shape[1]) if i not in binary_indices]
    x_cont_original = x_original[:, continuous_indices]
    x_cont_reconstructed = x_reconstructed[:, continuous_indices]

    # Continuous loss (using MSE)
    continuous_loss = F.mse_loss(x_cont_reconstructed, x_cont_original, reduction='mean')

    # Combine losses
    total_loss = alpha * binary_loss + (1 - alpha) * continuous_loss
    return total_loss

# Assume 'pca_df' contains your PCA results and you add 'labels' to this DataFrame
def plot_TSNE(df, labels = None, dist_matrix = None):
    if labels is None:
        labels = np.ones(df.shape[0])
    # Applying t-SNE directly to your dataset
    if dist_matrix is None:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    else :
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, metric="precomputed", init='random')
    tsne_results = tsne.fit_transform(dist_matrix)
    
    # Create a DataFrame to store results of t-SNE
    tsne_df = pd.DataFrame(data = tsne_results, columns = ['TSNE1', 'TSNE2'])
    tsne_df['labels'] = labels
    # Create a scatter plot with hue based on labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='labels', palette='Spectral', alpha=0.5)
    plt.title('t-SNE Visualization of Dataset with Labels')
    plt.show()


# Assume 'pca_df' contains your PCA results and you add 'labels' to this DataFrame
def plot_PCA(df, labels = None):
    if labels is None:
        labels = np.ones(df.shape[0])
    # Assume df is your DataFrame containing only numerical features
    pca = PCA(n_components=2)  # Reduce data to two dimensions if more than two
    principal_components = pca.fit_transform(df)
    
    # Create a new DataFrame for the PCA results
    pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    pca_df['labels'] = labels
    # Create a scatter plot with hue based on labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='labels', palette='Set2', alpha=0.5)
    plt.title('PCA of Dataset with Labels')
    plt.show()
    