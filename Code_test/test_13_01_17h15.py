from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Classe pour extraire les caractéristiques ROI.
    Elle sélectionne uniquement les 284 premières colonnes.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]  # ROI = premières colonnes

class VBMFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Classe pour extraire les caractéristiques VBM.
    Elle sélectionne les colonnes restantes après les caractéristiques ROI.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, 284:]  # VBM = colonnes après les 284 premières

def get_estimator():
    """
    Crée et retourne le pipeline combinant ROI et VBM.
    - Suppression des caractéristiques constantes pour chaque type de données.
    - Normalisation séparée pour ROI et VBM.
    - Sélection des caractéristiques avec SelectKBest pour ROI et VBM.
    - Fusion des caractéristiques combinées.
    - Régression logistique avec validation croisée.
    """
    # Pipeline pour les ROI
    roi_pipeline = make_pipeline(
        VarianceThreshold(threshold=0.0),  # Supprime les constantes
        MinMaxScaler(),                    # Normalisation
        SelectKBest(score_func=f_classif, k=75)  # Sélection des meilleures caractéristiques ROI
    )

    # Pipeline pour les VBM
    vbm_pipeline = make_pipeline(
        VarianceThreshold(threshold=0.0),  # Supprime les constantes
        MinMaxScaler(),                    # Normalisation
        SelectKBest(score_func=f_classif, k=100)  # Sélection des meilleures caractéristiques VBM
    )

    # Fusion des caractéristiques ROI et VBM
    combined_features = FeatureUnion([
        ('roi_features', make_pipeline(ROIsFeatureExtractor(), roi_pipeline)),
        ('vbm_features', make_pipeline(VBMFeatureExtractor(), vbm_pipeline))
    ])

    # Pipeline final
    return make_pipeline(
        combined_features,
        LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1, 10, 100],  # Plage pour régularisation
            cv=5,                 # Validation croisée à 5 plis
            penalty='l2',          # Régularisation L2
            scoring='roc_auc',     # Optimisation sur ROC-AUC
            solver='lbfgs',        # Solver adapté pour les données denses
            max_iter=1000,         # Convergence assurée
            random_state=42,
            n_jobs=-1
        )
    )
