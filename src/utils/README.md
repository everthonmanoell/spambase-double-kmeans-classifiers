# Utilitários da Questão 2

## Versões do dataset `spambase`

As duas funções de carregamento de dados devem disponibilizar as versões do
dataset exigidas no cabeçalho da Questão 2:

1. **Dataset original:** a variável resposta deve manter as 2 classes
	originais.
2. **Dataset com classes baseadas nos clusters:** a variável resposta deve
	ter um número de classes igual ao número de clusters de objetos `K*`
	obtido na Questão 1.

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


## Decisões de Arquitetura: Análise Estatística (Friedman e Nemenyi)

- **Fluxo Condicional (Omnibus e Post-hoc):** A implementação consolida o Teste de Friedman e o pós-teste de Nemenyi em uma única função estruturada (`perform_friedman_nemenyi`). Essa decisão garante a aderência à metodologia estatística correta: o teste *post-hoc* de Nemenyi só é acionado se, e somente se, o teste *omnibus* de Friedman rejeitar a hipótese nula global de equivalência entre os modelos ($p < \alpha$, com o padrão $\alpha = 0.05$). Isso impede programaticamente a execução de comparações múltiplas inválidas quando não há diferença global significativa.
- **Abstração e Generalização por Métrica:** Para atender à exigência de comparar os classificadores usando cada uma das métricas (Taxa de Erro, Precisão, Cobertura, F-measure), a função foi desenhada de forma agnóstica. Ela recebe um dicionário onde as chaves são os modelos e os valores são as distribuições de resultados ($300$ observações provenientes dos $30 \times 10$-folds). Essa abstração evita duplicação de código.
- **Validação de Entrada e Estruturação de Dados:** A função aplica *fail-fast*, verificando se todos os classificadores possuem o mesmo número exato de execuções (*folds*) antes de processar as estatísticas.
- **Integração Scipy e Scikit-Posthocs:** O cálculo da estatística de Friedman é delegado à função nativa `scipy.stats.friedmanchisquare`. Para o cálculo de Nemenyi, introduziu-se a biblioteca `scikit-posthocs`. Como esta biblioteca requer um padrão de dados tabular rigoroso, a nossa função encapsula e automatiza a transposição das listas de resultados nativas do Python para um `pandas.DataFrame` estruturado com o *shape* `(n_folds, n_classifiers)`, abstraindo a complexidade da integração.