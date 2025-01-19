from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]  # Limiter aux 284 premières colonnes

def get_estimator():
    estimator = make_pipeline(
        ROIsFeatureExtractor(),
        MinMaxScaler(),
        VotingClassifier(
            estimators=[
                ('logreg', LogisticRegressionCV(
                    Cs=10, cv=5, penalty='l2', solver='liblinear', class_weight='balanced'
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5, 
                    min_samples_leaf=2, class_weight='balanced', random_state=42
                ))
            ],
            voting='soft'
        )
    )
    return estimator