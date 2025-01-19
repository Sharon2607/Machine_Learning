from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
        GradientBoostingClassifier(
            n_estimators=200,  # Meilleur nombre d'arbres trouvé
            learning_rate=0.1,  # Meilleur taux d'apprentissage
            max_depth=3,  # Meilleure profondeur d'arbre
            min_samples_split=2,  # Meilleur nombre minimal d'échantillons pour diviser un nœud
            min_samples_leaf=10,  # Meilleur nombre minimal d'échantillons par feuille
            random_state=42
        )
    )
    return estimator
