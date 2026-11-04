# Spambase — Double K-Means & Classificadores Supervisionados (Projeto AM 2026-2)

Implementação experimental completa do projeto da disciplina de **Aprendizagem de Máquina (2026.2)** do **Centro de Informática (CIn/UFPE)**, ministrada pelo **Prof. Francisco de A. T. de Carvalho**, utilizando a base de dados **Spambase** ($4.601$ exemplos e $57$ atributos contínuos).

---

## 📄 Relatórios e Análises Concluídas

- **[`REPORT.md`](REPORT.md):** Relatório Técnico consolidado (Questões 1 e 2, metodologia, decisões de implementação, tabelas, gráficos e conclusões).
- **[`analysis/Q1_ANALYSIS.md`](analysis/Q1_ANALYSIS.md):** Análise detalhada da Questão 1 (Double K-Means, seleção de $(K^*, H^*) = (2, 2)$ por Silhueta, interpretação semântica dos blocos, matriz de protótipos $\mathbf{G}$, contingência e $\text{ARI}$).
- **[`analysis/Q2_STATISTICAL_ANALYSIS.md`](analysis/Q2_STATISTICAL_ANALYSIS.md):** Análise estatística da Questão 2 (Validação cruzada aninhada $30 \times 10$-folds, intervalos de confiança de $95\%$, teste de Friedman, pós-teste de Nemenyi e hiperparâmetros selecionados).
- **[`analysis/Q2_LEARNING_CURVES.md`](analysis/Q2_LEARNING_CURVES.md):** Curvas de aprendizagem de $5\%$ a $95\%$ (Treino vs. Teste) para os 5 classificadores nos dois datasets.

---

## 📊 Resumo dos Principais Resultados

### Questão 1 — Double K-Means ($900$ execuções)
- **Grade avaliada:** $K \in \{2, 3, 4\}$ e $H \in \{1, \dots, K\}$ ($9$ pares $\times 100$ inicializações = $900$ ajustes; $100\%$ convergidos).
- **Melhor configuração por Silhueta:** **$(K^*, H^*) = (2, 2)$** com **$\text{Sil} = 0{,}65963094$** e $W^* = 244.486{,}70$.
- **Índice de Rand Corrigido contra classes originais:** **$\text{ARI} = -0{,}00494970$** (o agrupamento não-supervisionado isola no Cluster 1 um grupo de $34$ e-mails legítimos com valores extremos nos $11$ atributos institucionais/telefônicos da HP Labs).

### Questão 2 — Classificação Supervisionada ($30 \times 10$-folds + 5-folds internos)

| Classificador | `original_2_classes` — Taxa de Erro (IC 95%) | `original_2_classes` — F1 Macro (IC 95%) | `kstar_classes` — Taxa de Erro (IC 95%) | `kstar_classes` — F1 Macro (IC 95%) |
| :--- | :---: | :---: | :---: | :---: |
| **GaussianBayes** | $0{,}1705\ [0{,}1701, 0{,}1709]$ | $0{,}8284\ [0{,}8280, 0{,}8288]$ | $0{,}0025\ [0{,}0024, 0{,}0026]$ | $0{,}8952\ [0{,}8887, 0{,}9017]$ |
| **BayesianKNN** | $0{,}0855\ [0{,}0846, 0{,}0863]$ | $0{,}9093\ [0{,}9084, 0{,}9102]$ | $0{,}0008\ [0{,}0007, 0{,}0008]$ | $0{,}9722\ [0{,}9699, 0{,}9745]$ |
| **ParzenWindow** | $0{,}1686\ [0{,}1676, 0{,}1697]$ | $0{,}8221\ [0{,}8209, 0{,}8232]$ | $0{,}0012\ [0{,}0011, 0{,}0013]$ | $0{,}9545\ [0{,}9502, 0{,}9588]$ |
| **LogisticRegression** | $0{,}0730\ [0{,}0725, 0{,}0734]$ | $0{,}9230\ [0{,}9226, 0{,}9235]$ | **$0{,}0008\ [0{,}0007, 0{,}0009]$** | **$0{,}9762\ [0{,}9723, 0{,}9802]$** |
| **MajorityVoting** | **$0{,}0716\ [0{,}0708, 0{,}0724]$** | **$0{,}9238\ [0{,}9229, 0{,}9246]$** | $0{,}0008\ [0{,}0007, 0{,}0009]$ | $0{,}9668\ [0{,}9634, 0{,}9702]$ |

---

## 🗂️ Estrutura do Repositório

```text
spambase-double-kmeans-classifiers/
├── dataset/
│   ├── spambase.data                 # Base de dados Spambase (4.601 × 58)
│   └── spambase.names                # Descrição dos 57 atributos e classes
├── docs/
│   ├── Projeto-AM-2026-2.pdf         # Especificação oficial do projeto
│   └── Double k-means.pdf            # Referência teórica do Double K-Means
├── src/
│   ├── clustering/
│   │   └── double_kmeans.py          # Algoritmo Double K-Means (Vichi, 2001)
│   ├── classification/
│   │   ├── bayes_gaussian.py         # Bayesiano Gaussiano Multivariado (MV + Cholesky)
│   │   ├── bayesian_knn.py           # Bayesiano k-Vizinhos (Euclidiana, City-Block, Chebyshev)
│   │   ├── parzen_window.py          # Bayesiano Janela de Parzen (Kernel Gaussiano Produto)
│   │   ├── custom_logistic_regression.py # Regressão Logística L2 com StandardScaler
│   │   └── majority_voting.py        # Ensemble por Regra do Voto Majoritário (Hard Voting)
│   ├── experiments/
│   │   ├── question_1.py             # Execução das 900 rodadas do Double K-Means
│   │   ├── question_2.py             # Pipeline 30×10-folds aninhado (5-folds internos)
│   │   └── learning_curves.py        # Curvas de aprendizagem de 5% a 95% (passo de 5%)
│   ├── analysis/
│   │   ├── silhouette.py             # Cálculo da Silhueta e seleção de (K*, H*)
│   │   ├── clustering_evaluation.py  # Cálculo de ARI, matriz G, contingência e relatório Q1
│   │   └── statistical_analysis.py   # ICs 95%, Teste de Friedman, Nemenyi e boxplots Q2
│   ├── visualization/
│   │   └── q1_plots.py               # Gráficos da Questão 1
│   └── utils/
│       ├── dataloader.py             # Carregamento das 2 versões da base e splitters estratificados
│       └── statistical_tests.py      # Funções auxiliares para Friedman e Nemenyi
├── results/                          # CSVs, JSON e TXTs gerados pelos experimentos (Q1 e Q2)
├── plots/                            # Gráficos gerados em alta resolução (.png)
├── report_latex/                     # relatório no formato latex
├── analysis/                         # Relatórios analíticos detalhados em Markdown
├── tests/                            # Suíte de testes automatizados (pytest)
└── REPORT.md                         # Relatório técnico completo do projeto
```

---

## ⚙️ Instalação e Configuração

O projeto utiliza **Python $\ge 3.14$** e **Poetry** para gerenciamento de dependências:

```bash
# Instalar dependências do projeto
poetry install
```

---

## 🚀 Como Reproduzir os Experimentos

### 1. Questão 1 — Double K-Means, Silhueta e ARI

```bash
# 1.1 Executar os 900 ajustes do Double K-Means (9 pares (K, H) × 100 repetições)
poetry run python -m src.experiments.question_1

# 1.2 Calcular a Silhueta dos 9 vencedores e selecionar (K*, H*)
poetry run python -m src.analysis.silhouette

# 1.3 Gerar ARI, matriz de protótipos G, matriz de confusão, gráficos e analysis/Q1_ANALYSIS.md
poetry run python -m src.analysis.clustering_evaluation
```

*(Para um teste rápido de fumaça com 2 execuções por par, use `poetry run python -m src.experiments.question_1 --n-runs 2 --output-dir results/q1_smoke`).*

---

### 2. Questão 2 — Validação Cruzada $30 \times 10$, Testes Estatísticos e Curvas de Aprendizagem

```bash
# 2.1 Validação cruzada estratificada 30×10 com tuning interno de 5-folds (paralelizado na CPU)
# Salva os 3.000 resultados em results/q2/cv_30x10_results.csv
poetry run python -m src.experiments.question_2 --n-jobs -1

# 2.2 Calcular ICs 95%, Teste de Friedman, Pós-teste de Nemenyi e gerar plots/comparison_boxplots.png
poetry run python -m src.analysis.statistical_analysis

# 2.3 Gerar curvas de aprendizagem de 5% a 95% (passo 5%) e gráficos em plots/
poetry run python -m src.experiments.learning_curves --n-jobs -1
```

---

## 🧪 Executando os Testes Automatizados

```bash
# Executar os testes unitários e de integração rápida
poetry run pytest tests/ -k "not TestQuestion2Pipeline" -v

# Executar a suíte completa (inclui teste end-to-end do pipeline 30×10)
poetry run pytest tests/ -v
```
