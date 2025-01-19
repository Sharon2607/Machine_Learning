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

def get_estimator():
    """
    Crée et retourne le pipeline basé uniquement sur ROI.
    - Suppression des caractéristiques constantes.
    - Normalisation des données.
    - Sélection des meilleures caractéristiques avec SelectKBest.
    - Régression logistique avec validation croisée.
    """
    # Pipeline pour les ROI
    roi_pipeline = make_pipeline(
        ROIsFeatureExtractor(),
        VarianceThreshold(threshold=0.0),  # Supprime les constantes
        MinMaxScaler(),                    # Normalisation
        SelectKBest(score_func=f_classif, k=75)  # Sélection des meilleures caractéristiques ROI
    )

    # Pipeline final
    return make_pipeline(
        roi_pipeline,
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
