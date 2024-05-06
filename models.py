import gower

# DBSCAN with Gower distance

    dist_matrix = gower.gower_matrix(dataset)
    db = DBSCAN(eps = 0.04, min_samples = 2, metric = "precomputed").fit(dist_matrix)
    labels = db.labels_
    print("Number of clusters: ", len(set(labels)) - (1 if -1 in labels else 0))
    print("Founded", len(np.where(labels == -1)[0]), "outliers")
    
    # TSNE plot
    TSNEPlot(dist_matrix, 'DBSCANGower.png', labels)
    
    # Print silhouette score
    silhouette = silhouette_score(dist_matrix, labels, metric='precomputed')
    print("Silhouette score: ", silhouette)
    
    # call the gridSearchDBSCAN method to find the best parameters for the DBSCAN clustering algorithm
    # eps = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    # min_samples = [2, 3, 4, 5, 6, 7, 9, 11, 15, 20]
    # best_eps, best_min_samples, best_score = gridSearchDBSCAN(dist_matrix, min_samples, eps)
    # print(best_eps, best_min_samples, best_score)
    
    
    # mi sono fidata ciecamente di cosa hai fatto al dataset
    dataset = pd.read_csv('dataset.csv', sep = ";", decimal=",")
    dataset = dataset.iloc[:, 1:-2]
    
    # One-Class SVM
    svm = OneClassSVM(nu=0.007) # is nu is the portion of anomalies
    outliers = svm.fit_predict(dataset)
    
    # flaggedDataset = dataset.copy()
    # flaggedDataset['outliers'] = outliers # mi sono fidata ciecamente anche qua di te
    # dist_matrix = gower.gower_matrix(dataset)
    # tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=300, metric="precomputed", init='random')
    # tsne_results = tsne.fit_transform(dist_matrix)
    # sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=outliers, palette=PAL)
    # plt.grid()
    # plt.show()
    
    numero_outliers = (outliers == -1).sum() # unica idea che ho avuto
    numero_non_outliers = (outliers == 1).sum()
    
    print("Numero di outliers:", numero_outliers)
    print("Numero di non outliers:", numero_non_outliers)
    
    print(adjusted_rand_score(outliers, labels))
    
    svmOutliers = np.where(outliers == -1)
    dbscanOutliers = np.where(labels == -1)
    svmOutliers = svmOutliers[0]
    dbscanOutliers = dbscanOutliers[0]
    
    counter = 0
    for i in range(len(svmOutliers)):
      for j in range(len(dbscanOutliers)):
        if(svmOutliers[i] == dbscanOutliers[j]):
          counter += 1
    print("Found ", counter, "common outliers between SVM and DBSCAN")