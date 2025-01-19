from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
        RandomForestClassifier(
            n_estimators=50,  # Meilleur nombre d'arbres trouvé
            max_depth=5,  # Meilleure profondeur trouvée
            min_samples_leaf=5,  # Meilleur nombre minimal d'échantillons par feuille
            random_state=42,
            class_weight="balanced",  # Pour gérer les classes déséquilibrées
            n_jobs=-1  # Utilisation maximale des cœurs disponibles
        )
    )
    return estimator
