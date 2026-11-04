# Utilitários da Questão 2

## Versões do dataset `spambase`

Três funções carregam dados em diferentes configurações:

1. **`load_spambase_original()`:** Retorna (X, y) onde y contém as 2 classes originais (0=não-spam, 1=spam).

2. **`load_spambase_kstar_mock()`:** Retorna (X, y_mock) onde y_mock é uma versão aleatória (descontinuada). **Use apenas para testes isolados que não dependem da Q1.**

3. **`load_spambase_kstar_real()`** (novo): Retorna (X, y_kstar) onde y_kstar contém os rótulos do clustering descoberto na Q1 (K*=2, H*=2). 
   - Lê os rótulos de `results/q1_object_labels.csv` (exato, reprodutível, não re-ajustado).
   - Valida metadados contra `results/q1_selection.json`.
   - Levanta `FileNotFoundError` se a Q1 não foi executada ainda.
   - **Esta é a versão correta para a Q2**, pois usa a partição real descoberta na Q1.

### Diferenças observadas

| Aspecto | Original | K*-Clustering |
|---------|----------|---------------|
| **Tamanho Classe 0** | 2.788 (60.6%) | 4.567 (99.3%) |
| **Tamanho Classe 1** | 1.813 (39.4%) | 34 (0.7%) |
| **Balanceamento** | Moderado | Muito desbalanceado |
| **Alinhamento com Q1** | — | ARI = −0.005 (negativo) |
| **Uso na Q2** | Baseline (referência) | Teste (com clustering) |

## Particionador de validação cruzada

A função `get_nested_cv_splitter` deve configurar a avaliação geral conforme o
item "a" da Questão 2:

- validação cruzada estratificada com `30 x 10 folds` para a avaliação geral;
- validação cruzada estratificada com `5 folds` nos 9 folds restantes para o
  ajuste de hiperparâmetros.

## Divisões para a curva de aprendizagem

A função `get_learning_curve_splits` deve gerar divisões estratificadas para o
item "d" da Questão 2. Os conjuntos de treinamento e teste devem variar nos
seguintes pares de proporções:

- de `(5%, 95%)` até `(95%, 5%)`;
- passos de `5%` entre cada divisão;
- amostragem estratificada em todas as divisões.

Execute `poetry run python -m src.experiments.learning_curves` para ajustar os
cinco classificadores nos dois datasets em cada tamanho, com dez repetições, e
salvar os dados por split, o resumo média/desvio-padrão e dois gráficos. Os
quatro classificadores base selecionam hiperparâmetros por CV estratificada
interna com F1 macro; MajorityVoting usa esses parâmetros nos membros do
ensemble. A curva plota F1 macro de treino/teste com faixa de ±1 desvio-padrão;
veja
[Q2_LEARNING_CURVES.md](../../analysis/Q2_LEARNING_CURVES.md).


## Decisões de Arquitetura: Análise Estatística (Friedman e Nemenyi)

- **Fluxo Condicional (Omnibus e Post-hoc):** A função `perform_friedman_nemenyi` consolida o teste de Friedman e o pós-teste de Nemenyi. O *post-hoc* só é executado se o teste *omnibus* rejeitar a hipótese nula global de equivalência entre os modelos ($p < \alpha$, padrão $\alpha = 0.05$), o que evita comparações múltiplas sem respaldo global. Como o Nemenyi é mais conservador que o Friedman, é possível que o Friedman rejeite $H_0$ e nenhum par apareça como significativo na matriz de p-valores. Nesse caso, o resultado é reportado como tal: há diferença global, mas ela não é atribuível a um par específico.

- **Definição dos blocos:** Cada bloco do Friedman é um par `(repetição, fold)`. Com $30 \times 10$-folds, temos $N = 300$ blocos, e cada linha do `DataFrame` de entrada corresponde a um deles. A ordem das linhas deve ser idêntica para todos os classificadores (recomenda-se um `MultiIndex` `(repetição, fold)` para garantir o alinhamento).

- **Limitação: blocos não independentes:** Os 300 blocos **não são independentes**, pois os conjuntos de treino se sobrepõem entre folds e repetições. O teste de Friedman assume blocos independentes, e, conforme Demšar (2006, §3.2.3), não há teste conhecido que corrija essa dependência. Consequência: os p-valores tendem a ser **otimistas**, e as conclusões valem para esta base de dados, não generalizam para outros domínios.

- **Interpretação com $N = 300$ e $k = 5$:** A Diferença Crítica (CD) do Nemenyi é $CD = q_\alpha \sqrt{\frac{k(k+1)}{6N}} \approx 0{,}352$ de rank médio. Por ser tão pequena, quase qualquer diferença será estatisticamente significativa. Por isso, o relatório apresenta, junto aos p-valores, a **diferença média da métrica com seu intervalo de confiança**, para distinguir significância estatística de relevância prática.

- **Abstração por Métrica:** A função é agnóstica à métrica (Taxa de Erro, Precisão, Cobertura, F-measure). Ela recebe um `pandas.DataFrame` com shape `(n_blocos, n_classificadores)`, em que cada coluna é um modelo e cada linha é um bloco, e o nome da métrica é usado apenas no log. Isso evita duplicação de código.

- **Validação de Entrada (*fail-fast*):** A função rejeita entradas que não sejam `DataFrame` (`TypeError`), que contenham NaN/infinito, que estejam vazias, que tenham menos de dois blocos, colunas repetidas ou `alpha` fora de `(0, 1)`. O Friedman exige ao menos 3 classificadores.

- **Integração Scipy e Scikit-Posthocs:** A estatística de Friedman é delegada a `scipy.stats.friedmanchisquare` e o Nemenyi a `scikit-posthocs` (`posthoc_nemenyi_friedman`), que devolve a matriz de p-valores par a par. Quando o Friedman não rejeita $H_0$, o campo `nemenyi_p_values` do resultado é `None`.

- **Retorno:** `perform_friedman_nemenyi` retorna `FriedmanNemenyiResult` com a estatística, p-valor e matriz de Nemenyi; a matriz é `None` quando o teste global não é significativo.

- **Execução Q2:** Rode `python -m src.analysis.statistical_analysis` para calcular ICs de 95% usando as médias das 30 repetições, imprimir métricas e testes no terminal, e salvar os arquivos descritos em [Q2_STATISTICAL_ANALYSIS.md](../../analysis/Q2_STATISTICAL_ANALYSIS.md). Para o Friedman, os 300 folds externos continuam sendo os blocos alinhados, portanto a dependência entre folds deve ser considerada na interpretação.