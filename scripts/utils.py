from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch

## 3d plots
import plotly.graph_objects as go



## globals
binary_indices = [i for i in range(1, 16)]
continuous_indices = [0] + [i for i in range(16, 21)]
alpha = len(binary_indices) / len(continuous_indices + binary_indices)

def dist(x, y): ## x :numpy array y: numpy array -> d : float
    cd = np.mean(np.abs(x.iloc[continuous_indices] - y.iloc[continuous_indices]))
    bd = np.sum(x.iloc[binary_indices] != y.iloc[binary_indices])
    d = alpha * cd + (1 - alpha) * bd
    return d

def gower_dist(x, y, binary_indices = binary_indices, continuous_indices = continuous_indices) -> float:
    alpha = len(binary_indices)/(len(x))
    x_bin = (x[binary_indices] >= 0.5).astype(int)
    y_bin = (y[binary_indices] >= 0.5).astype(int)
    # Binary loss (counting differing elements)
    binary_loss = (np.abs(x_bin - y_bin)).astype(float).mean()

    # Continuous data handling
    # continuous_indices = [i for i in range(len(x)) if i not in binary_indices]

    # Continuous data handling
    x_cont = x[continuous_indices]
    y_cont = y[continuous_indices]

    # Continuous loss (using Mean Absolute Error)
    continuous_loss = np.mean(np.abs(x_cont - y_cont))

    # Combine losses
    distance = alpha * binary_loss + (1 - alpha) * continuous_loss
    return distance

def plot_TSNE_2(df=None, labels=None, dist_matrix=None):
    if df is None and dist_matrix is None:
        print('Error: Either df or dist_matrix must be provided')
        return

    if labels is None:
        try:
            labels = np.ones(df.shape[0])
        except:
            labels = np.ones(dist_matrix.shape[0])

    if dist_matrix is not None:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30,
                    n_iter=1000, metric="precomputed", init='random',
                    random_state=42)
        tsne_results = tsne.fit_transform(dist_matrix)
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000,
                    random_state=42)
        tsne_results = tsne.fit_transform(df)

    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['labels'] = labels

    plt.figure(figsize=(20, 12))
    scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['labels'], cmap='viridis', alpha=0.5)

    plt.colorbar(scatter, label='Probability of an observation being an outlier')
    plt.title('t-SNE Visualization of Dataset with Labels')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.show()

def plot_TSNE(df=None, labels=None, dist_matrix=None):
    if df is None and dist_matrix is None:
        print('Error: Either df or dist_matrix must be provided')
        return

    if  labels is None:
        try : labels = np.ones(df.shape[0])
        except: labels = np.ones(dist_matrix.shape[0])

    if dist_matrix is not None:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, metric="precomputed", init='random',
                    random_state=42)
        tsne_results = tsne.fit_transform(dist_matrix)
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(df)

    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['labels'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue=tsne_df['labels'].astype(str), palette='cubehelix',
                    alpha=0.5)
    plt.title('t-SNE Visualization of Dataset with Labels')
    plt.show()


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



def gower_loss(x_original, x_reconstructed, binary_indices, continuous_indices) -> float:
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

    # Combine losses:
    total_loss = alpha * binary_loss + (1 - alpha) * continuous_loss
    return total_loss




def plot_3d_PCA(df, labels = None):
    if labels is None:
        labels = np.ones(df.shape[0])
    # Assume df is your DataFrame containing only numerical features
    pca = PCA(n_components=3)  # Reduce data to two dimensions if more than two
    principal_components = pca.fit_transform(df)

    # Create a new DataFrame for the PCA results
    pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2', 'PC3'])
    pca_df['labels'] = labels
    # Create a scatter plot with hue based on labels
    fig = go.Figure(data=[go.Scatter3d(
        x=pca_df['PC1'],
        y=pca_df['PC2'],
        z=pca_df['PC3'],
        mode='markers',
        marker=dict(
            size=6,
            color=pca_df['labels'], # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

## 3d TSNE
def plot_3d_TSNE(df=None, labels=None, dist_matrix=None):
    if df is None and dist_matrix is None:
        print('Error: Either df or dist_matrix must be provided')
        return

    if labels is None:
        try: labels = np.ones(df.shape[0])
        except: labels = np.ones(len(dist_matrix))

    if dist_matrix is not None:
        tsne = TSNE(n_components=3, verbose=1, perplexity=30, n_iter=1000, metric="precomputed", init='random')
        tsne_results = tsne.fit_transform(dist_matrix)
    else:
        tsne = TSNE(n_components=3, verbose=1, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(df)

    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    tsne_df['labels'] = labels

    fig = go.Figure(data=[go.Scatter3d(
        x=tsne_df['TSNE1'],
        y=tsne_df['TSNE2'],
        z=tsne_df['TSNE3'],
        mode='markers',
        marker=dict(
            size=6,
            color=tsne_df['labels'],  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])
    fig.show()

def sigmoid_to_prob(k=6) -> callable: ## closure parametrized by a k, number of methods
    # return a sigmoid-like function that takes number of methods that classify a single instance
    # as an outlier, and returns a probability.
    h = k/2
    def inner(x) -> float:
        return (((k* (1 + np.exp(-h)) * (1 + np.exp(h)))/(1 + np.exp(h) -(1 + np.exp(-h))))/
        (1 + np.exp(-x + h))) - k * ( 1 + np.exp(-h)) / (1 + np.exp(h) - (1 + np.exp(-h)))
    return inner

def jaccard_index(anom_label=-1) -> callable:
    ## closure parametrized by a label:
    # takes 2 lists of labels and compares the prediction of the anomalies using
    ## jaccard index, measure of comparison of 2 sets
    def inner(l1,l2) -> float:
        set1 = set(i for (i, label) in enumerate(l1) if label == anom_label)
        set2 = set(i for (i, label) in enumerate(l2) if label == anom_label)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 1.0 if not intersection else 0.0 # 1 if both are empty, else 0 if only one is empty
        return len(intersection) / len(union)
    return inner
