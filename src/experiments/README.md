## Decisões de Arquitetura: Experimento da Questão 1

* **Execução:** O script `q1_double_kmeans.py` roda a Questão 1 inteira e deve ser chamado a partir da raiz do repositório, porque os caminhos de `dataset/` e `results/` são relativos a ela:

  ```bash
  python -m src.experiments.q1_double_kmeans
  ```

  Ele usa a classe `DoubleKMeans` (`src/clustering/double_kmeans.py`), em que cada `fit` é uma única execução a partir de uma inicialização aleatória. Por isso, as repetições, a escolha da melhor execução e todas as métricas ficam neste script, como prevê a seção *Uma Execução por `fit`* de `src/clustering/README.md`.

* **100 Execuções por Par e Escolha pelo Menor $W$:** Para cada $K \in \{2, 3, 4\}$ e $H \in \{1, \dots, K\}$ (9 pares, 900 execuções), a função `run_pair` ajusta o double k-means 100 vezes e guarda a execução de menor `objective_`. Os slides (p. 7) mandam fazer exatamente isso, porque o algoritmo para num mínimo local que depende da inicialização. A troca da melhor execução exige melhora estrita (`<`), então, em caso de empate em $W$, fica a execução mais antiga. O `doublekm` do drclust faz o mesmo: só troca a melhor execução quando o critério melhora estritamente.

* **Sementes e Reprodutibilidade:** Um `np.random.default_rng(RANDOM_STATE)`, com `RANDOM_STATE = 42` (o mesmo valor do `dataloader`), sorteia uma semente por execução, na ordem dos pares: as 100 primeiras vão para $(2, 1)$, as 100 seguintes para $(2, 2)$ e assim por diante. Assim, o experimento inteiro depende de um único número. As sementes ficam no intervalo $[0, 2^{31} - 2]$, aceito pelo `RandomState` que a classe usa via `check_random_state`. Todas são gravadas em `runs.csv`, e a da melhor execução de cada par também vai para `summary.csv`. Para refazer só uma execução, sem repetir as 900, basta reajustar a classe com a semente gravada.

* **Silhueta:** Para cada par, a silhueta é `silhouette_score(X, best.row_labels_)`. Ela usa a partição de objetos da melhor execução, a distância euclidiana e todos os 4601 objetos (`sample_size=None`), sobre a mesma matriz $X$ do double k-means. É o mesmo cálculo da função `silhouette` do drclust. Duas propriedades devem constar no relatório:
  * a silhueta só depende da partição dos objetos e não enxerga $H$. Pares com o mesmo $K$ só diferem na silhueta porque $H$ muda a partição de objetos encontrada, e, se dois pares chegarem à mesma partição, a silhueta empata exatamente;
  * ela mede a separação entre objetos nas 57 variáveis originais, e não o critério $W$, que mede a distância de cada $x_{ij}$ ao protótipo do seu bloco.

* **Escolha de $(K^*, H^*)$:** O enunciado define $K^* = \arg\max_{(K,H)} Sil((K, H))$. Como o máximo é tomado sobre o par, o script escolhe o par $(K^*, H^*)$ e usa a melhor execução desse par como "o melhor resultado segundo a função objetivo com $K^*$". É a partir dela que saem o ARI, a matriz $G$, a matriz de confusão e o gráfico de $W$. A outra leitura possível seria pegar a execução de menor $W$ entre todos os $H$ com $K^*$, mas ela cairia sempre em $H = K^*$, porque o $W$ ótimo não aumenta com $H$ (dividir um grupo de variáveis nunca piora o ajuste). Por isso não foi adotada.
  * **Em aberto:** a regra de desempate, que fica na função `select_best_pair`. Desempatar pelo menor $W$ não serve, pelo mesmo motivo: favoreceria sempre o maior $H$.
  * **Limitação:** na literatura de partição two-mode, critérios que levam em conta a complexidade do modelo se saíram melhor que a silhueta na escolha do número de grupos, e a silhueta tendeu a subestimá-lo (Schepers, Ceulemans & Van Mechelen, 2008). O script segue a silhueta porque o enunciado a exige, e o relatório deve registrar essa limitação.

* **Índice de Rand Corrigido:** `adjusted_rand_score(y, best.row_labels_)` compara a partição de objetos de $K^*$ com as 2 classes a priori (Hubert & Arabie, 1985). O índice não depende da numeração dos grupos e funciona com $K^* \neq 2$. Para o comentário pedido no enunciado, há um ponto importante: o ARI só vale 1 quando as duas partições são idênticas. Com $K^* > 2$ grupos contra 2 classes isso é impossível, então o ARI fica abaixo de 1 mesmo que cada grupo contenha só spam ou só não spam. A matriz de confusão mostra se é esse o caso.

* **Matriz de Protótipos ($G$) e Grupos de Variáveis:** A matriz $G$ ($K^* \times H^*$) é gravada com as linhas $P_1, \dots, P_K$ (grupos de objetos) e as colunas $Q_1, \dots, Q_H$ (grupos de variáveis), na notação dos slides e dos comentários de `double_kmeans.py`. Como $g_{kh}$ é a média de um bloco, $G$ só se lê junto com as variáveis de cada $Q_h$. Por isso, `variable_groups.csv` lista as 57 variáveis com o seu grupo. Os nomes vêm das linhas `nome: continuous.` de `dataset/spambase.names`. A numeração dos grupos é arbitrária, porque vem da inicialização aleatória.

* **Matriz de Confusão:** A tabela grupos × classes é feita com `pd.crosstab`, que gera $K^* \times 2$. Não foi usada a `confusion_matrix` do scikit-learn porque ela trabalha com a união dos rótulos verdadeiros e previstos: com $K^* = 3$, por exemplo, ela devolveria uma matriz $3 \times 3$ com uma linha de zeros. A `sklearn.metrics.cluster.contingency_matrix` daria o mesmo resultado que o `crosstab`.

* **Gráfico da Função Objetivo por Iteração:** O gráfico usa o `objective_history_` da execução escolhida. A iteração 0 é a inicialização $(G^{(0)}, U^{(0)}, V^{(0)})$, na notação dos slides. Como cada passo minimiza $W$ em relação a um bloco de parâmetros (slides, p. 5–6), a curva não pode subir.
  * **Em aberto:** se a execução escolhida parar pelo `max_iter` sem convergir, o último ponto da curva pode ficar acima de `objective_` (ver o atributo `objective_history_` em `src/clustering/README.md`). Por enquanto, o script só imprime um aviso nesse caso.

* **Dados na Escala Original:** O script carrega a base com `load_spambase_original` e não transforma os dados, seguindo a decisão da seção *Dados na Escala Original* de `src/clustering/README.md`.
  * **Em aberto para discussão no grupo:** a literatura indica o contrário. O `doublekm` do drclust padroniza os dados por padrão (z-score), e Milligan & Cooper (1988) recomendam padronizar as variáveis antes do agrupamento.
  * **Por que isso importa na Spambase:** segundo a documentação da UCI, os desvios-padrão de `capital_run_length_total`, `capital_run_length_longest` e `capital_run_length_average` são 606,35, 194,89 e 31,73. Só a primeira variável responde por cerca de 90% da soma das variâncias, e as três juntas por quase toda ela. Na escala original, portanto, $W$, a silhueta e a partição de objetos devem ser determinadas quase só por essas variáveis. Isso é inferido das estatísticas publicadas, não foi medido. Com $H = 1$ o efeito é direto: $G$ tem uma única coluna, e cada objeto vai para o grupo cujo protótipo está mais perto da média da sua linha.
  * **Efeito na Questão 2:** a escolha afeta também a Questão 2, porque essa partição vira a variável resposta da segunda versão da base.

* **Saídas:** Tudo é gravado em `results/q1/`:

  | Arquivo | Conteúdo |
  |---|---|
  | `runs.csv` | as 900 execuções: $K$, $H$, semente, $W$, `n_iter`, `converged` e `n_relocations` |
  | `summary.csv` | um par por linha: $W$ e silhueta da melhor execução, semente, convergência e quantas das 100 execuções não convergiram |
  | `silhouette_vs_kh.png` | gráfico Sil × $(K, H)$, com o par escolhido em destaque |
  | `prototypes_G.csv` | item i: a matriz $G$ de $(K^*, H^*)$ |
  | `variable_groups.csv` | o grupo $Q_h$ de cada variável |
  | `confusion_matrix.csv` | item ii: grupos × classes (`spam` / `non-spam`) |
  | `objective_vs_iteration.png` | item iv: $W$ por iteração da execução escolhida |
  | `selected_run.json` | $K^*$, $H^*$, semente, $W$, silhueta, ARI e diagnósticos da execução escolhida |
  | `row_labels_kstar.csv` | a partição de objetos de $K^*$, na ordem das linhas de `dataset/spambase.data` |

  O enunciado lista os itens i, ii e iv; não há item iii.

* **Ligação com a Questão 2:** A segunda versão da base, com $K^*$ classes, usa como variável resposta a partição de objetos da execução escolhida, gravada em `row_labels_kstar.csv`. Esse arquivo deve substituir os rótulos aleatórios de `load_spambase_kstar_mock` (`src/utils/dataloader.py`). Gravar a partição evita repetir as 900 execuções a cada rodada da Questão 2.

* **Gráficos:** Os títulos e os eixos estão em português, para irem direto para o relatório e os slides. Cada gráfico tem uma única série, por isso não há legenda. No gráfico da silhueta, o par escolhido aparece em azul e os demais em cinza, e cada barra mostra o seu valor.

### Referências

* Vichi, M. (2001). *Double k-means clustering for simultaneous classification of objects and variables.* In: Borra, S., Rocci, R., Vichi, M., Schader, M. (eds.), *Advances in Classification and Data Analysis*, Springer, pp. 43–52. DOI 10.1007/978-3-642-59471-7_6.
* Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53–65. DOI 10.1016/0377-0427(87)90125-7.
* Hubert, L. & Arabie, P. (1985). *Comparing partitions.* Journal of Classification, 2(1), 193–218. DOI 10.1007/BF01908075.
* Milligan, G. W. & Cooper, M. C. (1988). *A study of standardization of variables in cluster analysis.* Journal of Classification, 5(2), 181–204. DOI 10.1007/BF01897163.
* Schepers, J., Ceulemans, E. & Van Mechelen, I. (2008). *Selecting among multi-mode partitioning models of different complexities: a comparison of four model selection criteria.* Journal of Classification, 25(1), 67–85. DOI 10.1007/s00357-008-9005-9.
* Prunila, I. & Vichi, M. (2025). *drclust: Simultaneous Clustering and (or) Dimensionality Reduction*, versão 0.1.1. Pacote R, CRAN. DOI 10.32614/CRAN.package.drclust. Funções `doublekm` (`src/double_km.cpp`, `src/prep.cpp`) e `silhouette` (`R/silhouette.R`).
* scikit-learn 1.9.1: `sklearn/metrics/cluster/_unsupervised.py` (`silhouette_score`), `sklearn/metrics/cluster/_supervised.py` (`adjusted_rand_score`, `contingency_matrix`) e `sklearn/metrics/_classification.py` (`confusion_matrix`). https://github.com/scikit-learn/scikit-learn
* Hopkins, M., Reeber, E., Forman, G. & Suermondt, J. (1999). *Spambase* (documentação com as estatísticas por atributo). UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION
