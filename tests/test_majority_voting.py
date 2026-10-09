import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.dataloader import load_spambase_original
from src.classification.majority_voting import MajorityVotingClassifier
from src.classification.bayes_gaussian import GaussianBayesClassifier
from src.classification.bayesian_knn import BayesianKNNClassifier
from src.classification.custom_logistic_regression import CustomLogisticRegression
from src.classification.parzen_window import ParzenWindowClassifier 


def evaluate_performance(y_true, y_pred, subset_name):
    """Calcula e exibe métricas de avaliação do Scikit-Learn."""
    acc = accuracy_score(y_true, y_pred)
    # average='macro' garante o cálculo correto mesmo quando a base tiver K* classes
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"--- Métricas no conjunto de {subset_name} ---")
    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")


def test_majority_voting():
    """Test the MajorityVotingClassifier with optimized base models."""
    print("=== TESTANDO CLASSIFICADOR DE VOTO MAJORITÁRIO ===")
    print("[INFO] A carregar dataset spambase...")
    
    X, y = load_spambase_original("dataset/spambase.data")
    
    # Divisão 70/30 estratificada para o teste rápido
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("[INFO] A otimizar hiperparâmetros dos classificadores base (GridSearch CV=3)...")
    
    # 1. Bayes Gaussiano
    grid_gauss = GridSearchCV(GaussianBayesClassifier(), GaussianBayesClassifier.get_param_grid(), cv=3, scoring='f1_macro', n_jobs=-1)
    grid_gauss.fit(X_train, y_train)
    
    # 2. Bayes KNN
    grid_knn = GridSearchCV(BayesianKNNClassifier(), BayesianKNNClassifier.get_param_grid(), cv=3, scoring='f1_macro', n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    
    # 3. Parzen Window
    grid_parzen = GridSearchCV(ParzenWindowClassifier(), ParzenWindowClassifier.get_param_grid(), cv=3, scoring='f1_macro', n_jobs=-1)
    grid_parzen.fit(X_train, y_train)
    
    # 4. Regressão Logística
    grid_logreg = GridSearchCV(CustomLogisticRegression(), CustomLogisticRegression.get_param_grid(), cv=3, scoring='f1_macro', n_jobs=-1)
    grid_logreg.fit(X_train, y_train)

    print("[INFO] Melhores hiperparâmetros encontrados. A instanciar o Voto Majoritário...")
    # Estrutura exigida pelo VotingClassifier com os estimadores já otimizados
    estimators = [
        ('gaussian', grid_gauss.best_estimator_),
        ('knn', grid_knn.best_estimator_),
        ('parzen', grid_parzen.best_estimator_),
        ('log_reg', grid_logreg.best_estimator_)
    ]

    ensemble = MajorityVotingClassifier(estimators=estimators)
    ensemble.fit(X_train, y_train)


    # 1) A CV interna está otimista em relação ao teste?
    for name, grid in [('gaussian', grid_gauss), ('knn', grid_knn),
                    ('parzen', grid_parzen), ('log_reg', grid_logreg)]:
        print(f"{name:9s} F1 CV interna = {grid.best_score_:.4f}")

    # 2) O k=1 ganha por muito ou por pouco?
    res = pd.DataFrame(grid_knn.cv_results_)[['param_metric', 'param_n_neighbors', 'mean_test_score']]
    print(res.sort_values('mean_test_score', ascending=False).head(10))

    # 3) Duplicatas favorecem k=1 e h pequeno
    print("Linhas duplicadas:", pd.DataFrame(X).duplicated().sum())

    # 4) O wrapper está votando corretamente? (empate 2-2 -> classe 0)
    votes = np.column_stack([est.predict(X_test) for _, est in estimators])
    manual = (votes.sum(axis=1) > 2).astype(int)
    print("Voto manual == ensemble:", np.array_equal(manual, ensemble.predict(X_test)))
    tie = votes.sum(axis=1) == 2
    print("Fração de spam real entre os empates:", (np.asarray(y_test)[tie] == 1).mean())

    votes = np.column_stack([est.predict(X_test) for _, est in estimators])
    n_ties = np.sum(votes.sum(axis=1) == 2)  # binário: 2 votos na classe 1 = empate 2-2
    print(f"Empates 2-2 no teste: {n_ties} de {len(y_test)} ({n_ties/len(y_test):.1%})")
    
    print(f"[OK] Classes mapeadas pelo ensemble: {ensemble.classes_}\n")

    print("[INFO] A gerar predições e avaliar desempenho...")
    preds_train = ensemble.predict(X_train)
    preds_test = ensemble.predict(X_test)
    
    # Avaliação completa de métricas em Treino e Teste
    evaluate_performance(y_train, preds_train, "Treino")
    evaluate_performance(y_test, preds_test, "Teste")


if __name__ == "__main__":
    test_majority_voting()