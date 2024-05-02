
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Assume 'pca_df' contains your PCA results and you add 'labels' to this DataFrame
def plot_TSNE(df, labels = None):
    if labels is None:
        labels = np.ones(df.shape[0])
    # Applying t-SNE directly to your dataset
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(df)
    
    # Create a DataFrame to store results of t-SNE
    tsne_df = pd.DataFrame(data = tsne_results, columns = ['TSNE1', 'TSNE2'])
    tsne_df['labels'] = labels
    # Create a scatter plot with hue based on labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='labels', palette='Set2', alpha=0.5)
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