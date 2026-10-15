import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from src.utils.dataloader import load_spambase_original
from src.classification.parzen_window import ParzenWindowClassifier

def test_parzen():
    print("Carregando dataset spambase...")
    X, y = load_spambase_original("dataset/spambase.data")
    
    # Divisão simples 70/30 para testar o classificador isolado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("\n--- 1. Teste de Fit e Predict Direto ---")
    clf = ParzenWindowClassifier(bandwidth=1.5)
    clf.fit(X_train, y_train)
    
    print(f"Probabilidades a priori calculadas: {clf.priors_}")
    
    preds = clf.predict(X_test)
    print(f"Previsões: {preds}")
    print(f"Valores reais: {y_test}")
    acuracia = np.mean(preds == y_test)
    print(f"Acurácia inicial (sem otimização): {acuracia:.4f}")

    print("\n--- 2. Teste de Compatibilidade com GridSearchCV ---")
    # Reduzindo os dados só para o teste rodar rápido
    X_subset, y_subset = X_train[:400], y_train[:400]
    
    param_grid = {'bandwidth': [0.1, 1.0, 5.0]}
    
    # n_jobs=-1 usa todos os núcleos da sua CPU para rodar as dobras em paralelo
    grid = GridSearchCV(
        ParzenWindowClassifier(), 
        param_grid, 
        cv=3, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    print("Iniciando busca em grade...")
    grid.fit(X_subset, y_subset)
    print(f"Busca finalizada! Melhor bandwidth encontrado no subset: {grid.best_params_}")
    print(f"Melhor F1-macro no subset: {grid.best_score_:.4f}")

if __name__ == "__main__":
    test_parzen()