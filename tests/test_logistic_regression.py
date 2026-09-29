import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from src.utils.dataloader import load_spambase_original
from src.classification.custom_logistic_regression import CustomLogisticRegression

def test_custom_logistic_regression():
    """Test the CustomLogisticRegression class with the Spambase dataset."""
    print("=== TESTANDO REGRESSÃO LOGÍSTICA CUSTOMIZADA ===")
    print("[INFO] Carregando dataset spambase...")
    
    # Carrega os dados reais do projeto
    X, y = load_spambase_original("dataset/spambase.data")
    
    # Divisão 70/30 estratificada apenas para o teste isolado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("\n--- 1. Teste de Fit e Predict Direto ---")
    clf = CustomLogisticRegression()
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acuracia = np.mean(preds == y_test)
    print(f"[OK] Acurácia inicial (sem otimização): {acuracia:.4f}")
    print(f"     Classes encontradas pelo modelo: {clf.classes_}")

    print("\n--- 2. Teste de Ajuste de Hiperparâmetros (CV Interna) ---")
    # Puxa a grade diretamente do método estático da nossa classe
    param_grid = CustomLogisticRegression.get_param_grid()
    
    # Configura o GridSearchCV simulando a etapa interna exigida no projeto (5-folds)
    grid = GridSearchCV(
        CustomLogisticRegression(), 
        param_grid, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    print("[INFO] Iniciando busca em grade (5-folds)...")
    grid.fit(X_train, y_train)
    
    print(f"[OK] Busca finalizada!")
    print(f"     Melhores hiperparâmetros: {grid.best_params_}")
    print(f"     Melhor F1-macro no treino: {grid.best_score_:.4f}")
    
    # Testa as predições com o melhor modelo encontrado
    best_preds = grid.predict(X_test)
    best_acc = np.mean(best_preds == y_test)
    print(f"[OK] Acurácia no teste com o melhor modelo: {best_acc:.4f}")

if __name__ == "__main__":
    test_custom_logistic_regression()