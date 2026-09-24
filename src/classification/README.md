## Decisões de Arquitetura e Hiperparâmetros: Regressão Logística

### 1. Arquitetura e Encapsulamento (Wrapper)
* **Padronização de Interface:** A classe `CustomLogisticRegression` foi construída como um *wrapper* (encapsulamento) herdando de `BaseEstimator` e `ClassifierMixin`. Essa decisão garante que o modelo nativo do Scikit-Learn tenha exatamente a mesma assinatura e comportamento das classes construídas do zero (como a Janela de Parzen). Isso mantém o script principal do experimento limpo e modular.
* **Isolamento da Grade de Busca:** A implementação do método `@staticmethod get_param_grid()` acopla o dicionário de hiperparâmetros diretamente à classe do modelo. Isso evita a poluição do script principal com dicionários soltos e garante que a configuração de busca pertença exclusivamente ao classificador correspondente.

### 2. Tratamento de Convergência e Escala
* **Ajuste de Iterações (`max_iter=10000`):** O conjunto de dados *Spambase* possui 57 atributos contínuos em escalas numéricas muito discrepantes (frequências percentuais versus contagens de comprimento de sequências de letras maiúsculas). O limite padrão de 100 iterações do Scikit-Learn é insuficiente para o algoritmo convergir para o mínimo global nesse cenário. O aumento para 10000 iterações como padrão melhora a possibilidade de convergência matemática correta e evita interrupções por avisos (*warnings*) de limite excedido durante o treinamento.

### 3. Definição da Grade de Busca (Hyperparameter Grid)
* **Força de Regularização (`C`):** A grade escolhida foi `[0.01, 0.1, 1.0, 10.0, 100.0]`. Utilizou-se uma escala logarítmica para explorar diferentes ordens de grandeza da regularização inversa. O espaço de busca cobre desde um modelo altamente penalizado e protegido contra *overfitting* (`0.01`) até um modelo com flexibilidade quase total para se ajustar aos dados de treinamento (`100.0`).
* **Penalidade (`penalty='l2'`) e Otimizador (`solver='lbfgs'`):** A penalidade L2 (Ridge) foi fixada por ser a mais robusta e numericamente estável para problemas com múltiplas variáveis contínuas. O otimizador `lbfgs` foi escolhido por ser o algoritmo padrão e mais eficiente em termos de memória para funções com penalidade L2. 

### 4. Dimensionamento do Custo Computacional
* **Viabilidade da Validação Cruzada Aninhada:** A metodologia do projeto exige uma validação cruzada externa de 30 repetições de 10-folds, com um loop interno de 5-folds nos dados de treino para o ajuste dos hiperparâmetros.
* **Controle de Explosão Combinatória:** Essa estrutura aninhada gera 1500 execuções base para cada modelo ($30 \times 10 \times 5$). Ao fixar o otimizador e o tipo de penalidade, limitando a grade a 5 valores do parâmetro `C`, o total de treinamentos da Regressão Logística foi contido em 7.500 execuções. A inclusão de outras penalidades (como L1) exigiria a troca de *solvers* e dobraria ou triplicaria o tamanho da grade, tornando o custo computacional inviável para o escopo do experimento.