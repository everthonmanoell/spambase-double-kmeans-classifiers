import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from src.utils.dataloader import load_spambase_original
from src.classification.bayesian_knn import BayesianKNNClassifier

def test_bayesian_knn():
    """Test the BayesianKNNClassifier class with the Spambase dataset."""
    print("=== TESTANDO CLASSIFICADOR BAYESIANO K-NN ===")
    print("[INFO] A carregar dataset spambase...")
    
    # Carrega os dados reais do projeto
    X, y = load_spambase_original("dataset/spambase.data")
    
    # Divisão 70/30 estratificada apenas para o teste isolado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("\n--- 1. Teste de Fit e Predict Direto ---")
    clf = BayesianKNNClassifier(n_neighbors=5, metric='euclidean')
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acuracia = np.mean(preds == y_test)
    print(f"[OK] Acurácia inicial (k=5, euclidean): {acuracia:.4f}")
    print(f"     Classes encontradas pelo modelo: {clf.classes_}")

    print("\n--- 2. Teste de Ajuste de Hiperparâmetros (CV Interna) ---")
    # Puxa a grelha diretamente do método estático da nossa classe
    param_grid = BayesianKNNClassifier.get_param_grid()
    
    # Configura o GridSearchCV simulando a etapa interna (5-folds)
    grid = GridSearchCV(
        BayesianKNNClassifier(), 
        param_grid, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    print("[INFO] A iniciar busca em grade (5-folds)...")
    grid.fit(X_train, y_train)
    
    print(f"[OK] Busca finalizada!")
    print(f"     Melhores hiperparâmetros: {grid.best_params_}")
    print(f"     Melhor F1-macro no treino: {grid.best_score_:.4f}")
    
    # Testa as predições com o melhor modelo encontrado
    best_preds = grid.predict(X_test)
    best_acc = np.mean(best_preds == y_test)
    print(f"[OK] Acurácia no teste com o melhor modelo: {best_acc:.4f}")

if __name__ == "__main__":
    test_bayesian_knn()