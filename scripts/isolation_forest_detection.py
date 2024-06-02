from sklearn.ensemble import IsolationForest
import importlib

# custom libraries
import utils
importlib.reload(utils) #debug - remove
import preprocessing
importlib.reload(preprocessing) #debug - remove

def main(dataframe):
    # contamination is the percentage of expected outliers
    isolation_forest = IsolationForest(contamination=0.05,
                                       bootstrap=True,
                                       random_state=123)
    # fit the model and get the predictions on the dataset
    outlier_labels = isolation_forest.fit_predict(dataframe)
    # prediction will contain 0 for inliers and -1 for outliers
    return outlier_labels

if __name__ == '__main__':
    main()