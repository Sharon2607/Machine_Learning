from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284] 

def get_estimator():
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        VotingClassifier(
            estimators=[
                ('svm', SVC(
                    C=1, kernel='rbf', probability=True, class_weight='balanced', gamma='scale', random_state=42
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    min_samples_leaf=2, class_weight='balanced', random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=3, 
                    min_samples_split=5, min_samples_leaf=2, random_state=42
                ))
            ],
            voting='soft'
        )
    )
    return estimator