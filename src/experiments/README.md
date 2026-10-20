## Decisões de Arquitetura: Experimento da Questão 1

* **Execução:** O script `q1_double_kmeans.py` roda a Questão 1 inteira e deve ser chamado a partir da raiz do repositório, porque os caminhos de `dataset/` e `results/` são relativos a ela:

  ```bash
  python -m src.experiments.q1_double_kmeans
  ```

  Ele usa a classe `DoubleKMeans` (`src/clustering/double_kmeans.py`), em que cada `fit` é uma única execução a partir de uma inicialização aleatória. Por isso, as repetições, a escolha da melhor execução e todas as métricas ficam neste script, como prevê a seção *Uma Execução por `fit`* de `src/clustering/README.md`.

* **100 Execuções por Par e Escolha pelo Menor $W$:** Para cada $K \in \{2, 3, 4\}$ e $H \in \{1, \dots, K\}$ (9 pares, 900 execuções), a função `run_pair` ajusta o double k-means 100 vezes e guarda a execução de menor `objective_`. Os slides (p. 7) mandam fazer exatamente isso, porque o algoritmo para num mínimo local que depende da inicialização. A troca da melhor execução exige melhora estrita (`<`), então, em caso de empate em $W$, fica a execução mais antiga. O `doublekm` do drclust faz o mesmo: só troca a melhor execução quando o critério melhora estritamente.

* **Sementes e Reprodutibilidade:** Um `np.random.default_rng(RANDOM_STATE)`, com `RANDOM_STATE = 42` (o mesmo valor do `dataloader`), sorteia uma semente por execução, na ordem dos pares: as 100 primeiras vão para $(2, 1)$, as 100 seguintes para $(2, 2)$ e assim por diante. Assim, o experimento inteiro depende de um único número. As sementes ficam no intervalo $[0, 2^{31} - 2]$, aceito pelo `RandomState` que a classe usa via `check_random_state`. Todas são gravadas em `runs.csv`, e a da melhor execução de cada par também vai para `summary.csv`. Para refazer só uma execução, sem repetir as 900, basta reajustar a classe com a semente gravada.

* **Silhueta:** Para cada par, a silhueta é `silhouette_score(X, best.row_labels_)`. Ela usa a partição de objetos da melhor execução, a distância euclidiana e todos os 4601 objetos (`sample_size=None`), sobre a mesma matriz padronizada usada no double k-means. É o mesmo cálculo da função `silhouette` do drclust. Duas propriedades devem constar no relatório:
  * a silhueta só depende da partição dos objetos e não enxerga $H$. Pares com o mesmo $K$ só diferem na silhueta porque $H$ muda a partição de objetos encontrada, e, se dois pares chegarem à mesma partição, a silhueta empata exatamente;
  * ela mede a separação entre objetos nas 57 variáveis padronizadas, e não o critério $W$, que mede a distância de cada $x_{ij}$ ao protótipo do seu bloco.

* **Escolha de $(K^*, H^*)$:** O enunciado define $K^* = \arg\max_{(K,H)} Sil((K, H))$. Como o máximo é tomado sobre o par, o script escolhe o par $(K^*, H^*)$ e usa a melhor execução desse par como "o melhor resultado segundo a função objetivo com $K^*$". É a partir dela que saem o ARI, a matriz $G$, a matriz de confusão e o gráfico de $W$. A outra leitura possível seria pegar a execução de menor $W$ entre todos os $H$ com $K^*$, mas ela cairia sempre em $H = K^*$, porque o $W$ ótimo não aumenta com $H$ (dividir um grupo de variáveis nunca piora o ajuste). Por isso não foi adotada.
  * **Desempate:** pares com o mesmo $K$ que chegam à mesma partição de objetos têm exatamente a mesma silhueta. Nesse caso, a função `select_best_pair` fica com o modelo mais simples: o menor $K$ e, depois, o menor $H$. É o princípio de parsimônia usado na escolha do número de grupos; a regra do gap statistic, por exemplo, escolhe o menor número de grupos dentro de um erro-padrão (Tibshirani, Walther & Hastie, 2001). Desempatar pelo menor $W$ não serve, pelo mesmo motivo do parágrafo acima: favoreceria sempre o maior $H$. A comparação é exata, como o arg max do enunciado: silhuetas apenas próximas não contam como empate, e a regra não depende da ordem das linhas do `summary`.
  * **Limitação:** numa simulação com partição two-mode, os critérios que levam em conta a complexidade do modelo (DIFFIT e convex hull) acertaram o número de grupos com muito mais frequência que a extensão two-mode da silhueta. Quando essa silhueta errou, ela sempre subestimou o número de grupos (Schepers, Ceulemans & Van Mechelen, 2008). A silhueta do estudo combina objetos e variáveis, enquanto a usada aqui considera só os objetos, então o resultado não se transfere diretamente, mas indica a limitação. O script segue a silhueta porque o enunciado a exige, e o relatório deve registrar essa limitação.

* **Índice de Rand Corrigido:** `adjusted_rand_score(y, best.row_labels_)` compara a partição de objetos de $K^*$ com as 2 classes a priori (Hubert & Arabie, 1985). O índice não depende da numeração dos grupos e funciona com $K^* \neq 2$. Para o comentário pedido no enunciado, há um ponto importante: o ARI só vale 1 quando as duas partições são idênticas. Com $K^* > 2$ grupos contra 2 classes isso é impossível, então o ARI fica abaixo de 1 mesmo que cada grupo contenha só spam ou só não spam. A matriz de confusão mostra se é esse o caso.

* **Matriz de Protótipos ($G$) e Grupos de Variáveis:** A matriz $G$ ($K^* \times H^*$) é gravada com as linhas $P_1, \dots, P_K$ (grupos de objetos) e as colunas $Q_1, \dots, Q_H$ (grupos de variáveis), na notação dos slides e dos comentários de `double_kmeans.py`. Como $g_{kh}$ é a média de um bloco, $G$ só se lê junto com as variáveis de cada $Q_h$. Com os dados padronizados, $g_{kh}$ está em desvios-padrão: um valor positivo indica que, em média, os objetos de $P_k$ estão acima da média geral nas variáveis de $Q_h$. Por isso, `variable_groups.csv` lista as 57 variáveis com o seu grupo. Os nomes vêm das linhas `nome: continuous.` de `dataset/spambase.names`. A numeração dos grupos é arbitrária, porque vem da inicialização aleatória.

* **Matriz de Confusão:** A tabela grupos × classes é feita com `pd.crosstab`, que gera $K^* \times 2$. Não foi usada a `confusion_matrix` do scikit-learn porque ela trabalha com a união dos rótulos verdadeiros e previstos: com $K^* = 3$, por exemplo, ela devolveria uma matriz $3 \times 3$ com uma linha de zeros. A `sklearn.metrics.cluster.contingency_matrix` daria o mesmo resultado que o `crosstab`.

* **Gráfico da Função Objetivo por Iteração:** O gráfico usa o `objective_history_` da execução escolhida. A iteração 0 é a inicialização $(G^{(0)}, U^{(0)}, V^{(0)})$, na notação dos slides. Como cada passo minimiza $W$ em relação a um bloco de parâmetros (slides, p. 5–6), a curva não pode subir. O último ponto é sempre o `objective_` reportado, mesmo quando a execução para pelo `max_iter` (ver o atributo `objective_history_` em `src/clustering/README.md`); nesse caso, o script só imprime um aviso.

* **Padronização das Variáveis (z-score):** Antes do double k-means, o script padroniza cada variável com o `StandardScaler` (média 0 e desvio-padrão 1). A classe continua sem transformar os dados (seção *Sem Transformação na Classe* de `src/clustering/README.md`). A decisão foi tomada pelo grupo, pelos motivos abaixo:
  * **A escala decide o peso de cada variável:** $W$ soma diferenças ao quadrado (slides, p. 4), então cada variável pesa na proporção da sua variância. Segundo a documentação da UCI, os desvios-padrão de `capital_run_length_total`, `capital_run_length_longest` e `capital_run_length_average` são 606,35, 194,89 e 31,73, enquanto as outras 54 variáveis são porcentagens. Só a primeira responde por cerca de 90% da soma das variâncias, e as três juntas por quase toda ela. Sem padronizar, $W$, a silhueta e a partição de objetos seriam determinadas quase só por essas variáveis. Isso é inferido das estatísticas publicadas, não foi medido.
  * **Sensibilidade do método:** os slides (p. 8) listam entre as desvantagens do double k-means a sensibilidade a valores discrepantes e a variáveis irrelevantes.
  * **Literatura:** o `doublekm` do drclust, pacote de Prunila & Vichi (este último é o autor do método), padroniza os dados por padrão (z-score). Milligan & Cooper (1988) compararam oito estratégias de padronização numa simulação: dividir pela amplitude da variável deu a melhor recuperação dos grupos, de forma consistente, e o z-score foi menos eficaz em várias situações. O grupo escolheu o z-score por ser o padrão do drclust. A alternativa apoiada por esse estudo é a padronização pela amplitude, que o drclust também oferece (`prep = 2`, min-max).
  * **Detalhe do cálculo:** o `StandardScaler` divide pelo desvio-padrão populacional ($N$). Dividir por $N - 1$ multiplicaria todas as variáveis pela mesma constante, o que não muda as partições nem a silhueta.
  * **Consequências:** $W$, a silhueta e $G$ ficam na escala padronizada, e a partição escolhida vira a variável resposta da segunda versão da base na Questão 2.

* **Saídas:** Tudo é gravado em `results/q1/`:

  | Arquivo | Conteúdo |
  |---|---|
  | `runs.csv` | as 900 execuções: $K$, $H$, semente, $W$, `n_iter`, `converged` e `n_relocations` |
  | `summary.csv` | um par por linha: $W$ e silhueta da melhor execução, semente, convergência e quantas das 100 execuções não convergiram |
  | `silhouette_vs_kh.png` | gráfico Sil × $(K, H)$, com o par escolhido em destaque |
  | `prototypes_G.csv` | item i: a matriz $G$ de $(K^*, H^*)$, na escala padronizada |
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
* Schepers, J., Ceulemans, E. & Van Mechelen, I. (2008). *Selecting among multi-mode partitioning models of different complexities: a comparison of four model selection criteria.* Journal of Classification, 25(1), 67–85. DOI 10.1007/s00357-008-9005-9. Versão pré-publicação em acesso aberto: Technical Report 0695, IAP Statistics Network, 2006 (§4.1.2). https://sites.uclouvain.be/IAP-Stat-Phase-V-VI/PhaseV/publications_2006/TR/TR0695.pdf
* Tibshirani, R., Walther, G. & Hastie, T. (2001). *Estimating the number of clusters in a data set via the gap statistic.* Journal of the Royal Statistical Society, Series B, 63(2), 411–423. DOI 10.1111/1467-9868.00293.
* Prunila, I. & Vichi, M. (2025). *drclust: Simultaneous Clustering and (or) Dimensionality Reduction*, versão 0.1.1. Pacote R, CRAN. DOI 10.32614/CRAN.package.drclust. Funções `doublekm` (`src/double_km.cpp`, `src/prep.cpp`) e `silhouette` (`R/silhouette.R`).
* scikit-learn 1.9.1: `sklearn/preprocessing/_data.py` (`StandardScaler`), `sklearn/metrics/cluster/_unsupervised.py` (`silhouette_score`), `sklearn/metrics/cluster/_supervised.py` (`adjusted_rand_score`, `contingency_matrix`) e `sklearn/metrics/_classification.py` (`confusion_matrix`). https://github.com/scikit-learn/scikit-learn
* Hopkins, M., Reeber, E., Forman, G. & Suermondt, J. (1999). *Spambase* [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C53G6X. Estatísticas por atributo na documentação: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION
