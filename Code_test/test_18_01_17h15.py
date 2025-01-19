from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]

def get_estimator():
    """Build your estimator here."""
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        VotingClassifier(
            estimators=[
                ('svm', SVC(
                    C=10,
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced',
                    gamma='auto',
                    random_state=42
                )),
                ('reglog', LogisticRegressionCV(
                    cv=5,
                    class_weight='balanced',
                    max_iter=1000,
                    Cs=20,
                    random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.01,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=4,
                    random_state=42
                ))
            ],
            voting='soft'
        )
    )
    return estimator
