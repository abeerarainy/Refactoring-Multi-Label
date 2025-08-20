from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

def perform_feature_reduction(X, y):
    estimator = SVC(kernel="linear")  # Use a classifier since y is categorical
    selector = RFECV(estimator, step=1, cv=5)
    selector.fit(X, y)
    return X[X.columns[selector.support_]]  # Return only selected features