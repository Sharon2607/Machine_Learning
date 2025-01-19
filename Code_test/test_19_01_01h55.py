from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]

def get_estimator():
    """Build your estimator here."""
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        LogisticRegressionCV(
            Cs=[0.1],  
            cv=5,
            penalty='l1',
            solver='saga',
            scoring='roc_auc',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  
        )
    )
    return estimator
