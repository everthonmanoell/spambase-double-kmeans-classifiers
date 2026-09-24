import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

class BayesianKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador bayesiano baseado em k-vizinhos[cite: 11].
    Encapsula o KNeighborsClassifier avaliando as distâncias Euclidiana, 
    City-Block e Chebishev[cite: 11].
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Treina o modelo baseado nos vizinhos mais próximos."""
        self.model_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        """Realiza a predição das classes."""
        return self.model_.predict(X)
        
    def predict_proba(self, X):
        """
        Retorna as probabilidades a posteriori de cada classe, 
        calculadas através da proporção matemática k_i / k.
        """
        return self.model_.predict_proba(X)

    @staticmethod
    def get_param_grid():
        """
        Retorna a grade de busca para fixar o número de vizinhos (k) 
        e a métrica de distância via validação cruzada[cite: 11].
        """
        return {
            'n_neighbors': [1, 3, 5, 7, 11, 15, 21, 31],
            'metric': ['euclidean', 'cityblock', 'chebyshev']
        }