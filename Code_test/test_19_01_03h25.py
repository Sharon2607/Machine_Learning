from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
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
    svm_model = SVC(kernel=correlation_kernel, C=1, probability=True, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    log_reg_cv_model = LogisticRegressionCV(Cs=10, cv=5, max_iter=1000, random_state=42)

    # Méta-modèle
    meta_model = LogisticRegression(random_state=42)

    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('svm', svm_model),
            ('gb', gb_model),
            ('log_reg_cv', log_reg_cv_model)
        ],
        final_estimator=meta_model,
        cv=5  # Validation croisée intégrée
    )

    # Pipeline
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        stacking_clf
    )

    return estimator