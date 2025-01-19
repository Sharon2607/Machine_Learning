from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import numpy as np

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
    """Créer une pipeline avec Voting Classifier optimisé par GridSearchCV."""
    svm_model = SVC(kernel=correlation_kernel, C=1, probability=True, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    log_reg_cv_model = LogisticRegressionCV(Cs=10, cv=5, max_iter=1000, random_state=42)

    # Voting Classifier avec meilleurs poids
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('gb', gb_model),
            ('log_reg_cv', log_reg_cv_model)
        ],
        voting='soft',
        weights=[3, 2, 1]  # Meilleurs poids
    )

    # Pipeline
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        voting_clf
    )

    return estimator