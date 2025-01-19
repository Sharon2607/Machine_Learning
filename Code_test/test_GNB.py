from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]
    
def correlation_kernel(X, Y=None):
    if Y is None:
        Y = X
    return np.corrcoef(X, Y)[:X.shape[0], X.shape[0]:]

def get_estimator():
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        GaussianNB()
    )
    return estimator
