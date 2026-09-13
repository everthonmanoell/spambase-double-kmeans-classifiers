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
