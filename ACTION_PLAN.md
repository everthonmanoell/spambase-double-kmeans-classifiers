# 📋 PLANO DE AÇÃO DETALHADO

## Fase 1: QUESTÃO 1 - Double K-Means (02-08 Nov)

### Tarefa 1.1: Script de Execução em Larga Escala (4-6 horas)

**Arquivo:** `src/experiments/question_1.py` (novo)

**O que fazer:**
```python
# Estrutura esperada:
1. Carregar dataset spambase
2. Para K em [2, 3, 4]:
   Para H em [1, 2, ..., K]:  # K=2→H∈{1,2}; K=3→H∈{1,2,3}; K=4→H∈{1,2,3,4}
     resultados_kh = []
     Para run em range(100):
       dkm = DoubleKMeans(n_row_clusters=K, n_col_clusters=H, random_state=run)
       dkm.fit(X)
       resultados_kh.append({
         'run': run,
         'K': K, 
         'H': H,
         'objetivo': dkm.objective_,
         'n_iter': dkm.n_iter_,
         'labels': dkm.labels_,
         'prototypes': dkm.prototypes_
       })
     
     # Selecionar melhor por função objetivo
     melhor = resultados_kh[argmin(objetivo)]
     salvar em CSV/JSON
```

**Saídas esperadas:**
- `results/q1_executions_summary.csv` - Uma linha por (K,H) com melhor objetivo
- `results/q1_full_history.pkl` - Histórico completo com todos os 900 resultados
- Estrutura: K, H, best_objective, best_run_id, n_iterations, convergence_rate

---

### Tarefa 1.2: Análise de Silhueta (2-3 horas)

**Arquivo:** `src/analysis/silhouette.py` (novo)

**O que fazer:**
1. Para cada par (K,H) - usar melhor resultado da Tarefa 1.1
2. Calcular silhueta usando `sklearn.metrics.silhouette_score(X, labels)`
3. Salvar em estrutura 2D (K × H)
4. Encontrar K* = argmax(silhueta)

```python
silhouette_matrix = np.zeros((3, 4))  # K∈{2,3,4}, H∈{1,2,3,4}
for K, H in todas_combinacoes:
  labels = melhor_resultado(K, H).labels_
  sil_score = silhouette_score(X, labels)
  silhouette_matrix[K-2, H-1] = sil_score

K_star = encontrar_argmax(silhouette_matrix)
```

**Saídas esperadas:**
- `results/silhouette_scores.csv` - Matriz K × H com scores
- `plots/silhouette_heatmap.png` - Heatmap mostrando K*
- Print: `K* = X` (valor encontrado)

---

### Tarefa 1.3: Matriz de Confusão e ARI (2-3 horas)

**Arquivo:** `src/analysis/clustering_evaluation.py` (novo)

**O que fazer:**
1. Para o melhor resultado com K* do Double K-Means:
   - Obter labels do clustering: `u_star = dkm.labels_`
   - Obter labels priori originais: `y` (2 classes do spambase)

2. Calcular:
   ```python
   # Matriz de confusão
   confusion_matrix(y_true=y, y_pred=u_star)
   
   # Índice de Rand Ajustado
   ari = adjusted_rand_index(y, u_star)
   ```

3. Comentar sobre resultados

**Saídas esperadas:**
- `plots/confusion_matrix_q1.png` - Matriz de confusão
- `results/ari_score.txt` - Valor de ARI com interpretação
- Comentários em um arquivo `analysis/Q1_ANALYSIS.md`

---

### Tarefa 1.4: Visualizações (2 horas)

**Arquivo:** `src/visualization/q1_plots.py` (novo)

**Plots requeridos:**

1. **Matriz de Protótipos (G)**
   ```python
   G = melhor_dkm.prototypes_  # shape: (K*, H*)
   plt.imshow(G, aspect='auto', cmap='viridis')
   plt.colorbar()
   plt.title(f'Block Prototypes Matrix G (K*={K*})')
   ```

2. **Função Objetivo vs Iterações**
   ```python
   plt.plot(melhor_dkm.objective_history_)
   plt.xlabel('Iteration')
   plt.ylabel('Objective Function W(G, U, V)')
   plt.title('Convergence of Double K-Means')
   ```

3. **Heatmap Silhueta** (já em 1.2)

**Saídas esperadas:**
- `plots/prototypes_heatmap.png`
- `plots/objective_convergence.png`
- `plots/silhouette_heatmap.png`

---

## Fase 2: QUESTÃO 2 - Classificação (09-15 Nov)

### Tarefa 2.1: Integração K* com Dataset (1-2 horas)

**Arquivo:** `src/utils/dataloader.py` (modificar)

**O que fazer:**
```python
def load_spambase_kstar_real(filepath="dataset/spambase.data", 
                              dkm_results_path="results/q1_executions_summary.csv"):
    """
    Carrega spambase com K* classes do Double K-Means,
    não mais com labels mock aleatórios.
    """
    X, _ = load_spambase_original(filepath)
    
    # Carregar melhor resultado com K*
    results = pd.read_csv(dkm_results_path)
    K_star = results['K'].iloc[argmax(silhouette)]
    
    # Carregar labels do clustering
    dkm_best = load_pickle(f"results/q1_best_dkm_K{K_star}.pkl")
    y_kstar = dkm_best.labels_
    
    return X, y_kstar
```

**Saídas esperadas:**
- Função pronta para usar em experimentos
- Confirmação: print com K*, distribuição de classes

---

### Tarefa 2.2: Pipeline de CV 30×10 (6-8 horas)

**Arquivo:** `src/experiments/question_2.py` (novo)

**Estrutura:**
```python
def run_30x10_cv_experiment():
    X, y_original = load_spambase_original()
    X, y_kstar = load_spambase_kstar_real()
    
    classifiers = {
        'bayes_gaussian': GaussianBayesClassifier(),
        'bayesian_knn': BayesianKNNClassifier(),
        'parzen': ParzenWindowClassifier(),
        'logistic': CustomLogisticRegression(),
        'majority_voting': MajorityVotingClassifier(...)
    }
    
    datasets = {
        'original_2_classes': (X, y_original),
        'kstar_classes': (X, y_kstar)
    }
    
    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
    
    results = []
    
    for dataset_name, (X, y) in datasets.items():
        for fold_idx, (train_outer, test_outer) in enumerate(outer_cv.split(X, y)):
            X_train_outer = X[train_outer]
            y_train_outer = y[train_outer]
            X_test = X[test_outer]
            y_test = y[test_outer]
            
            for clf_name, clf in classifiers.items():
                # Tuning internal
                best_params = grid_search_cv(
                    clf, clf.get_param_grid(), 
                    X_train_outer, y_train_outer, inner_cv
                )
                
                # Treinar com melhores parâmetros
                clf.set_params(**best_params)
                clf.fit(X_train_outer, y_train_outer)
                
                # Avaliar
                y_pred = clf.predict(X_test)
                
                metrics = {
                    'dataset': dataset_name,
                    'classifier': clf_name,
                    'fold': fold_idx,
                    'error': 1 - accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='macro'),
                    'recall': recall_score(y_test, y_pred, average='macro'),
                    'f1': f1_score(y_test, y_pred, average='macro'),
                    'best_params': best_params
                }
                results.append(metrics)
    
    return pd.DataFrame(results)
```

Para MajorityVoting, ajustar cada um dos quatro classificadores base na CV
interna com seus grids; depois, montar e ajustar a votação majoritária com os
melhores parâmetros de cada membro no conjunto de treino externo completo.
Não aplicar ao ensemble um grid vazio, pois isso deixaria todos os membros nos
valores iniciais.

Para `GaussianBayesClassifier`, estimar priores $P(\omega_i)=N_i/N$, médias
$\hat{\mu}_i$ e covariâncias $\hat{\Sigma}_i$ por Máxima Verossimilhança
(divisor $N_i$), sem hiperparâmetros de busca (`get_param_grid() == {}`). Para
garantir matriz definida positiva na fatoração de Cholesky quando $N_i \le d=57$
(classe minoritária de `kstar_classes`) ou quando atributos esparsos têm
variância zero em folds/subamostras, utilizar estabilização numérica diagonal
fixa `reg_covar=1e-6` ($\tilde{\Sigma}_i = \hat{\Sigma}_i + 10^{-6}I$).

**Saídas esperadas:**
- `results/cv_30x10_results.csv` - 30×10×2 datasets × 5 classifiers = 3000 linhas
- Colunas: fold_id, classifier, dataset, error, precision, recall, f1, hyperparams

---

### Tarefa 2.3: Análise Estatística (3-4 horas)

**Arquivo:** `src/analysis/statistical_analysis.py` (novo e completar `statistical_tests.py`)

**O que fazer:**

1. **Intervalos de Confiança (IC 95%)**
   ```python
   def compute_ci_95(metric_array):
       mean = np.mean(metric_array)
       se = np.std(metric_array) / np.sqrt(len(metric_array))
       ci_lower = mean - 1.96 * se
       ci_upper = mean + 1.96 * se
       return mean, ci_lower, ci_upper
   ```

2. **Friedman Test** (já iniciado)
   ```python
   # Para cada métrica e dataset:
   stat, p_value = friedmanchisquare(
       resultado_bayes,
       resultado_knn,
       resultado_parzen,
       resultado_logistic,
       resultado_voting
   )
   print(f"Friedman p-value: {p_value}")
   ```

3. **Nemenyi Post-hoc** (completar)
   ```python
   # Se Friedman significante (p < 0.05):
   p_matrix = sp.posthoc_nemenyi_friedmanchisquare(
       df_results
   )
   ```

**Saídas esperadas:**
- `results/confidence_intervals.csv`
- `results/friedman_results.txt`
- `results/nemenyi_pvalues.csv`
- `plots/comparison_boxplots.png`

---

### Tarefa 2.4: Curva de Aprendizagem (3-4 horas)

**Arquivo:** `src/experiments/learning_curves.py` (novo)

Para cada split, ajustar os quatro classificadores base por busca em grade com
CV estratificada no conjunto de treino daquele split (`f1_macro`). Usar até
cinco folds internos, reduzindo esse número para a quantidade da classe
minoritária quando ela for menor que cinco. Reajustar cada melhor estimador no
treino completo. Majority Voting deve usar os quatro classificadores com esses
hiperparâmetros selecionados; não há tuning independente para o ensemble.

**O que fazer:**
```python
def plot_learning_curves():
    splits = get_learning_curve_splits(X, y)  # Já existe
    
    training_sizes = sorted(splits.keys())  # [0.05, 0.10, ..., 0.95]
    
    for dataset_name in ['original_2_classes', 'kstar_classes']:
        X, y = carregar_dataset(dataset_name)
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for clf_idx, (clf_name, clf) in enumerate(classifiers.items()):
            train_scores = []
            test_scores = []
            
            for train_size in training_sizes:
                fold_scores_train = []
                fold_scores_test = []
                
                for train_idx, test_idx in splits[train_size]:
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_test = X[test_idx]
                    y_test = y[test_idx]
                    
                    best_params = grid_search_cv(
                        clf, clf.get_param_grid(), X_train, y_train, inner_cv
                    )
                    clf.set_params(**best_params)
                    clf.fit(X_train, y_train)
                    
                    f1_train = f1_score(y_train, clf.predict(X_train), average='macro')
                    f1_test = f1_score(y_test, clf.predict(X_test), average='macro')
                    
                    fold_scores_train.append(f1_train)
                    fold_scores_test.append(f1_test)
                
                train_scores.append(np.mean(fold_scores_train))
                test_scores.append(np.mean(fold_scores_test))
            
            ax = axes[clf_idx]
            ax.plot(training_sizes, train_scores, 'o-', label='Training')
            ax.plot(training_sizes, test_scores, 's-', label='Testing')
            ax.fill_between(training_sizes, 
                            np.array(train_scores) - np.std(fold_scores_train),
                            np.array(train_scores) + np.std(fold_scores_train),
                            alpha=0.2)
            ax.set_title(clf_name)
            ax.set_xlabel('Training Size')
            ax.set_ylabel('F-measure')
            ax.legend()
```

**Saídas esperadas:**
- `plots/learning_curves_original.png`
- `plots/learning_curves_kstar.png`
- `results/learning_curves_data.csv`
- `results/learning_curves_summary.csv` - médias e desvios-padrão das repetições
- Registrar parâmetros selecionados e folds internos por split no CSV detalhado

---

## Fase 3: DOCUMENTAÇÃO (16-18 Nov)

### Tarefa 3.1: Relatório Técnico (4-6 horas)

**Arquivo:** `REPORT.md` ou `REPORT.pdf`

**Seções:**
1. **Capa** - Título, autores, data, instituição
2. **Resumo Executivo** - 1 página
3. **Introdução** - Contexto, objetivo, motivação
4. **Dados** - Descrição spambase, pré-processamento
5. **Metodologia**
   - Questão 1: Double K-Means
   - Questão 2: Classificadores e CV
   - Questão 3: Testes estatísticos
6. **Resultados**
   - Tabelas com valores numéricos
   - Gráficos de todas as análises
   - Testes estatísticos
7. **Discussão** - Interpretação dos resultados
8. **Conclusão**
9. **Referências**

---

### Tarefa 3.2: Slides de Apresentação (3-4 horas)

**Formato:** PowerPoint, Google Slides ou PDF

**Slide Structure (10 minutos = ~15-20 slides):**
1. Capa (1 min)
2. Objetivo (1 min)
3. Dataset (1 min)
4. Metodologia Q1 (2 min)
5. Resultados Q1 (2 min)
6. Metodologia Q2 (2 min)
7. Resultados Q2 - Tabelas (2 min)
8. Resultados Q2 - Gráficos (2 min)
9. Testes Estatísticos (1 min)
10. Curva de Aprendizagem (1 min)
11. Conclusões (1 min)

---

### Tarefa 3.3: Preparação Entrega Google Classroom (1 hora)

**Arquivos individuais (NÃO .zip):**
- [ ] Código fonte: `src/` (todos os .py)
- [ ] Testes: `tests/` (todos os .py)
- [ ] Relatório: `REPORT.pdf`
- [ ] Slides: `PRESENTATION.pptx` (ou PDF)
- [ ] README com instruções de execução
- [ ] Resultados: `results/` (CSV com números principais)

---

## ⏱️ CRONOGRAMA RECOMENDADO

| Data | Tarefa | Horas | Responsável |
|------|--------|-------|------------|
| 02-03 Nov | 1.1: Script Q1 | 6 | ? |
| 04 Nov | 1.2: Silhueta | 3 | ? |
| 05 Nov | 1.3: ARI + Confusão | 3 | ? |
| 06-07 Nov | 1.4: Plots Q1 | 2 | ? |
| 08 Nov | 📊 Q1 PRONTO | - | ✅ |
| 09-11 Nov | 2.1-2.2: Pipeline CV | 8 | ? |
| 12 Nov | 2.3: Testes Estatísticos | 4 | ? |
| 13-14 Nov | 2.4: Curva Aprendizagem | 4 | ? |
| 15 Nov | 📊 Q2 PRONTO | - | ✅ |
| 16-17 Nov | 3.1: Relatório + Slides | 8 | ? |
| 18 Nov | 🎯 ENTREGA + APRESENTAÇÃO | - | ✅ |

---

## 🔍 Checkpoints de Qualidade

- [ ] Todos os scripts têm docstrings
- [ ] Outputs salvos em `results/` e `plots/`
- [ ] Valores numéricos confirmados manualmente
- [ ] Plots têm títulos, labels, legendas
- [ ] Relatório tem todas as 9 seções
- [ ] Slides legíveis e concisos
- [ ] README explica como reproduzir experimentos
- [ ] Código segue PEP 8

---

**Boa sorte! 💪**
