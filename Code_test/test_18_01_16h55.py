from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
 
class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y):
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
            ('svm', SVC(C=10, kernel='rbf', probability=True, class_weight='balanced', gamma='auto', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=1, class_weight='balanced', random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10, min_samples_leaf=1, random_state=42))
        ],
        voting='soft'
    )
)
    return estimator