## Decisões de Arquitetura: Double k-means

* **Algoritmo Seguido:** A classe `DoubleKMeans` implementa o double k-means dos slides da disciplina (`docs/Double k-means.pdf`), que obtém ao mesmo tempo uma partição exclusiva dos objetos em $K$ grupos, uma partição exclusiva das variáveis em $H$ grupos e a matriz de protótipos $G = (g_{kh})$ dos $K \times H$ blocos. A função objetivo minimizada é $W(G, U, V) = \sum_{k=1}^{K}\sum_{h=1}^{H}\sum_{i=1}^{N}\sum_{j=1}^{P} u_{ik}\, v_{jh}\, (x_{ij} - g_{kh})^2$. A otimização alterna os três passos dos slides, sempre nessa ordem, até que nenhum objeto e nenhuma variável mude de grupo:
  1. com $U$ e $V$ fixos, $g_{kh}$ é a média do bloco $kh$: $g_{kh} = \frac{\sum_i \sum_j u_{ik} v_{jh} x_{ij}}{n_k\, n_h}$;
  2. com $G$ e $V$ fixos, cada objeto vai para o grupo $k$ que minimiza $\sum_h \sum_j v_{jh} (x_{ij} - g_{kh})^2$;
  3. com $G$ e $U$ fixos, cada variável vai para o grupo $h$ que minimiza $\sum_k \sum_i u_{ik} (x_{ij} - g_{kh})^2$.

  Como nos slides, $G$ não é recalculado entre os passos 2 e 3. O parâmetro `max_iter` é apenas um limite de segurança; o atributo `converged_` indica se a parada ocorreu pelo critério dos slides.

* **Integração com o Ecossistema Scikit-Learn:** A classe herda de `ClusterMixin` e `BaseEstimator`, nessa ordem, pelo mesmo motivo explicado na seção do Bayesiano Gaussiano em `src/classification/README.md`. O `__init__` apenas guarda os parâmetros (`n_row_clusters` = $K$, `n_col_clusters` = $H$, `max_iter`, `random_state`), e a entrada é verificada com `validate_data`. Depois do `fit`, os resultados ficam nos atributos:
  * `row_labels_` (também em `labels_`) e `column_labels_`: as partições de objetos ($U$) e de variáveis ($V$);
  * `prototypes_`: a matriz $G$, de tamanho $K \times H$;
  * `objective_`: o valor final de $W$;
  * `objective_history_`: o valor de $W$ na inicialização e ao fim de cada iteração, usado no plot da função objetivo versus as iterações;
  * `n_iter_`, `converged_` e `n_relocations_`.

* **Uma Execução por `fit`:** Cada chamada ao `fit` é uma execução a partir de uma inicialização aleatória, controlada por `random_state`. As 100 execuções por par $(K, H)$ exigidas na Questão 1, e a escolha da execução de menor $W$, ficam a cargo do script do experimento, que também precisa dos resultados de cada execução.

* **Dados na Escala Original:** A classe não transforma os dados. O enunciado não prevê nenhuma transformação, e a escolha foi aplicar o double k-means à matriz original da Spambase.

* **Inicialização:** $U^{(0)}$ e $V^{(0)}$ são sorteados, atribuindo cada objeto e cada variável a um grupo aleatório. Para que nenhum grupo comece vazio, os $K$ (ou $H$) primeiros itens de uma permutação aleatória recebem um grupo diferente cada. $G^{(0)}$ é então obtido pelo passo 1.

* **Cálculo Matricial dos Passos:** Com $U$ ($N \times K$) e $V$ ($P \times H$) em formato *one-hot*, as somas dos blocos saem de $U^\top X V$. Os custos dos passos 2 e 3 saem da expansão $(x - g)^2 = x^2 - 2xg + g^2$, descartando o termo $x^2$, que não depende do grupo. Assim, cada passo é um pequeno número de produtos de matrizes, sem laços sobre objetos ou variáveis, o que importa para as 900 execuções da Questão 1.

* **Tratamento de Grupos Vazios:** Os slides não tratam desse caso. A inicialização garante grupos não vazios, mas os passos 2 e 3 podem esvaziar um grupo de objetos ($n_k = 0$) ou de variáveis ($n_h = 0$). Nesse caso, o passo 1 calcularia $0/0$, e o `NaN` resultante corromperia os passos seguintes sem gerar erro. A solução adotada é reposicionar: logo após os passos 2 e 3, cada grupo vazio recebe o objeto (ou a variável) de maior custo em relação ao próprio protótipo, escolhido entre os grupos com pelo menos dois membros. O protótipo do grupo que recebeu o item passa a ser a média desse item em cada grupo da outra partição.
  * **Justificativa:** o custo do item movido só pode diminuir, porque a média minimiza a soma de quadrados, e o grupo doador não fica vazio. Com isso, $W$ não aumenta entre iterações, e toda execução termina com exatamente $K$ grupos de objetos e $H$ grupos de variáveis. Nenhuma execução precisa ser descartada.
  * **Precedentes:** a mesma regra do "ponto de maior custo" é usada pelo `KMeans` do scikit-learn (`_relocate_empty_clusters_dense`, em `sklearn/cluster/_k_means_common.pyx`). A implementação do double k-means publicada por Prunila & Vichi, o pacote R `drclust` (função `doublekm`), também preenche o grupo vazio logo após a alocação em vez de descartar a execução, com uma variante mais cara: divide em dois o grupo mais heterogêneo.
  * O atributo `n_relocations_` registra quantos reposicionamentos ocorreram em cada execução. Por segurança, o passo 1 levanta um `RuntimeError` se ainda assim encontrar um bloco vazio.

### Referências

* Vichi, M. (2001). *Double k-means clustering for simultaneous classification of objects and variables.* In: Borra, S., Rocci, R., Vichi, M., Schader, M. (eds.), *Advances in Classification and Data Analysis*, Springer, pp. 43–52. DOI 10.1007/978-3-642-59471-7_6.
* Prunila, I. & Vichi, M. `drclust` 0.1.1, pacote R. https://cran.r-project.org/package=drclust
* scikit-learn, `sklearn/cluster/_k_means_common.pyx`. https://github.com/scikit-learn/scikit-learn
