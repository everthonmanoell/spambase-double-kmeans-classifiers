import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.dataloader import load_spambase_original
from src.classification.majority_voting import MajorityVotingClassifier

# Importação dos modelos base
from src.classification.bayes_gaussian import GaussianBayesClassifier
from src.classification.bayesian_knn import BayesianKNNClassifier
from src.classification.custom_logistic_regression import CustomLogisticRegression
# Assumindo o nome da classe Parzen que está no arquivo parzen_window.py[cite: 7]
from src.classification.parzen_window import ParzenWindowClassifier 

def test_majority_voting():
    """Test the MajorityVotingClassifier with the 4 base models."""
    print("=== TESTANDO CLASSIFICADOR DE VOTO MAJORITÁRIO ===")
    print("[INFO] A carregar dataset spambase...")
    
    X, y = load_spambase_original("dataset/spambase.data")
    
    # Divisão 70/30 estratificada para o teste rápido
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("[INFO] A inicializar os 4 classificadores base...")
    clf1 = GaussianBayesClassifier()
    clf2 = BayesianKNNClassifier(n_neighbors=5, metric='euclidean')
    clf3 = ParzenWindowClassifier()
    clf4 = CustomLogisticRegression()

    # Estrutura exigida pelo VotingClassifier: lista de tuplos (nome, objeto)
    estimators = [
        ('gaussian', clf1),
        ('knn', clf2),
        ('parzen', clf3),
        ('log_reg', clf4)
    ]

    print("[INFO] A instanciar o Voto Majoritário...")
    ensemble = MajorityVotingClassifier(estimators=estimators)

    print("[INFO] A treinar o ensemble (treina os 4 modelos internamente)...")
    ensemble.fit(X_train, y_train)
    print(f"[OK] Classes mapeadas pelo ensemble: {ensemble.classes_}")

    print("[INFO] A gerar predições por voto hard...")
    preds = ensemble.predict(X_test)
    
    acc = np.mean(preds == y_test)
    print(f"[OK] Acurácia final do Voto Majoritário: {acc:.4f}")

if __name__ == "__main__":
    test_majority_voting()