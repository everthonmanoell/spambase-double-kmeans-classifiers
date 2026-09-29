import numpy as np
from src.utils.dataloader import (
    load_spambase_original, 
    load_spambase_kstar_mock, 
    get_nested_cv_splitter, 
    get_learning_curve_splits
)
from src.classification.parzen_window import ParzenWindowClassifier

def main():
    print("=== TESTANDO DATA LOADER ===")
    try:
        # 1. Testando carregamento original
        X, y = load_spambase_original("dataset/spambase.data")
        print(f"[OK] Versão Original carregada. X shape: {X.shape}, y shape: {y.shape}")
        print(f"     Classes encontradas: {np.unique(y)}")

        # 2. Testando mock do K*
        X_mock, y_mock = load_spambase_kstar_mock("dataset/spambase.data", k_star=3)
        print(f"[OK] Versão Mock (K*=3) carregada. X shape: {X_mock.shape}, y shape: {y_mock.shape}")
        print(f"     Classes encontradas: {np.unique(y_mock)}")

        # 3. Testando splits de curva de aprendizagem
        splits = get_learning_curve_splits(X, y)
        train_5_idx, test_5_idx = splits[0.05]
        train_95_idx, test_95_idx = splits[0.95]
        print(f"[OK] Curva de aprendizagem gerada. Total de partições: {len(splits)}")
        print(f"     Tamanho do treino a 5%: {len(train_5_idx)} amostras")
        print(f"     Tamanho do treino a 95%: {len(train_95_idx)} amostras")

        # 4. Testando splits de CV aninhada
        cv = get_nested_cv_splitter()
        n_splits = cv.get_n_splits(X, y)
        print(f"[OK] Validação Cruzada 30x10 instanciada. Total de splits externos: {n_splits}")

    except Exception as e:
        print(f"[ERRO] Falha ao carregar ou processar dados: {e}")
        return

    print("\n=== TESTANDO JANELA DE PARZEN ===")
    try:
        # Pega apenas uma pequena amostra para o teste rodar instantaneamente
        X_train, y_train = X[:200], y[:200]
        X_test, y_test = X[-10:], y[-10:]

        clf = ParzenWindowClassifier(bandwidth=1.5)
        
        # Testando o fit
        clf.fit(X_train, y_train)
        print(f"[OK] Fit concluído. Priors calculados: {clf.priors_}")

        # Testando o predict
        preds = clf.predict(X_test)
        print(f"[OK] Predict concluído. Predições: {preds}")
        print(f"     Classes Reais  : {y_test}")

    except Exception as e:
        print(f"[ERRO] Falha na Janela de Parzen: {e}")

if __name__ == "__main__":
    main()