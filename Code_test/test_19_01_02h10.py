from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        SVC(
            C=1,  # Meilleur paramètre C
            kernel='rbf',  # Meilleur noyau
            gamma='scale',  # Meilleur gamma
            probability=True,  # Activer les probabilités pour ROC-AUC
            random_state=42
        )
    )
    return estimator
