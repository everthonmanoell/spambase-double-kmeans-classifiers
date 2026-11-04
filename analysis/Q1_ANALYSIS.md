# Análise da Questão 1 — Agrupamento Co-clustering com Double K-Means

## 1. Protocolo Experimental e Pré-processamento dos Dados

- **Base de dados Spambase:** $N = 4.601$ observações e $d = 57$ atributos contínuos; a última coluna contém a classe original a priori (`0` = não-spam, com $2.788$ exemplos; `1` = spam, com $1.813$ exemplos).
- **Padronização (*z-score*):** Como os 57 atributos possuem escalas muito discrepantes (frequências percentuais em $[0, 100]$ vs. contagens de letras maiúsculas na casa dos milhares), aplicou-se padronização *z-score* por atributo ($\mu_j = 0, \sigma_j = 1$) no conjunto de dados completo, preservando a ordem original das linhas. A variável resposta original não participa do ajuste nem da seleção por silhueta.
- **Formulação do Double K-Means:** O algoritmo particiona simultaneamente as $N$ linhas em $K$ clusters de objetos (matriz indicadora binária $\mathbf{U}_{N \times K}$) e as $d$ colunas em $H$ clusters de variáveis (matriz indicadora binária $\mathbf{V}_{d \times H}$), atualizando iterativamente a matriz de protótipos dos blocos $\mathbf{G}_{K \times H}$ para minimizar a função objetivo de soma de erros quadráticos intra-bloco:
  $$W(\mathbf{G}, \mathbf{U}, \mathbf{V}) = \sum_{i=1}^{N} \sum_{j=1}^{d} \sum_{k=1}^{K} \sum_{h=1}^{H} u_{ik} v_{jh} (x_{ij} - g_{kh})^2$$
- **Grade de busca e inicializações:** Executou-se o algoritmo **100 vezes** (sementes `0` a `99`) para cada um dos **9 pares $(K, H)$** com $K \in \{2, 3, 4\}$ e $H \in \{1, \dots, K\}$, totalizando **900 ajustes independentes**.
  - **Convergência:** Convergiram **900 de 900 ajustes** ($100\%$) dentro do limite máximo de $100$ iterações.
  - **Critério de seleção em duas etapas:**
    1. *Dentro de cada par $(K, H)$:* seleciona-se a repetição com o **menor valor da função objetivo $W$** (empates exatos resolvidos pela menor semente).
    2. *Entre os 9 pares $(K, H)$:* calcula-se o índice de **Silhueta ($\text{Sil}$)** da partição de objetos $\mathbf{U}$ de cada vencedor e escolhe-se $(K^*, H^*) = \arg\max_{(K, H)} \text{Sil}(K, H)$.
- **Hash SHA-256 do dataset:** `b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12`.

---

## 2. Seleção de $(K^*, H^*)$ por Função Objetivo e Silhueta

A tabela abaixo resume o melhor resultado (menor $W$ dentre as 100 execuções) e o respectivo índice de Silhueta para cada um dos 9 pares $(K, H)$:

| K | H | n_runs | best_run_id | best_objective | best_converged | silhouette |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 100 | 0 | 256513.74824370 | True | 0.29609380 |
| **2** | **2** | **100** | **0** | **244486.69834951** | **True** | **0.65963094** |
| 3 | 1 | 100 | 0 | 254414.01287916 | True | 0.25146824 |
| 3 | 2 | 100 | 82 | 239421.78540288 | True | 0.23673145 |
| 3 | 3 | 100 | 7 | 234585.86636711 | True | 0.14092129 |
| 4 | 1 | 100 | 0 | 253278.67951748 | True | 0.04842578 |
| 4 | 2 | 100 | 3 | 233358.70221378 | True | 0.24279690 |
| 4 | 3 | 100 | 8 | 228516.44366872 | True | 0.16196117 |
| 4 | 4 | 100 | 18 | 227250.87512132 | True | 0.17745596 |

- **Par selecionado:** **$K^* = 2, H^* = 2$ (semente `0`)**, com **$\text{Sil}(2, 2) = 0{,}65963094$** e função objetivo $W = 244.486{,}69834951$.
- **Análise do comportamento de $W$ vs. $\text{Sil}$:** Como esperado teoricamente, o valor ótimo da função objetivo $W$ decresce monotonamente conforme aumentamos os graus de liberdade $K$ e $H$ (chegando a $227.250{,}88$ em $K=4, H=4$), razão pela qual $W$ é usado apenas para comparar inicializações *dentro* de um mesmo par $(K, H)$. Para escolher a estrutura ótima entre diferentes dimensionalidades de blocos, o índice de Silhueta aponta de forma inequívoca $(K^*, H^*) = (2, 2)$, cujo score ($0{,}6596$) é mais que o dobro do segundo colocado ($K=2, H=1$ com $0{,}2961$). Quando $K \ge 3$, o algoritmo passa a fragmentar a nuvem principal de e-mails em subgrupos sobrepostos, derrubando a Silhueta para a faixa de $0{,}048$ a $0{,}251$.

![Silhueta](../plots/silhouette_heatmap.png)

---

## 3. Índice de Rand Corrigido (ARI) e Matriz de Confusão vs. Partição a Priori

Para a partição vencedora com $K^* = 2$ (e $H^* = 2$), comparou-se o agrupamento obtido com os rótulos originais da base Spambase (`0` = não-spam, `1` = spam):

- **Índice de Rand Corrigido:** **$\text{ARI} = -0{,}0049496989$**.

### Tabela de Contingência (Matriz de Confusão: Classes a Priori $\times$ Clusters Obtidos)

| original_class | 0 | 1 |
| --- | --- | --- |
| 0 | 2754 | 34 |
| 1 | 1813 | 0 |

*(Nota: Os IDs dos clusters são arbitrários em aprendizado não-supervisionado; não se interpreta a diagonal como acertos de classificação nem se renomeiam clusters como spam/não-spam).*

![Contingencia](../plots/confusion_matrix_q1.png)

### Distribuição dos Clusters de Objetos ($K^* = 2$)

| cluster | n_objects | fraction |
| --- | --- | --- |
| 0 | 4567 | 0.99261030 |
| 1 | 34 | 0.00738970 |

### Comentário Crítico sobre o ARI e a Estrutura Encontrada
1. **Por que o ARI é praticamente zero ($-0{,}00495$) se a Silhueta é alta ($0{,}6596$)?**
   O ARI mede a concordância entre duas partições descontando o acaso ($\text{ARI} = 1$ indica partições idênticas; $\text{ARI} \approx 0$ indica independência/nível do acaso; valores ligeiramente negativos ocorrem quando a concordância é inferior ao valor esperado sob hipergeométrica). A tabela de contingência revela o motivo: o Double K-Means colocou **$99{,}26\%$ da base ($4.567$ e-mails, incluindo todos os $1.813$ spams e $2.754$ não-spams) no Cluster 0**, e isolou apenas **$34$ e-mails ($0{,}74\%$, todos da classe não-spam) no Cluster 1**.
2. **Significado semântico dos 34 objetos do Cluster 1:**
   Inspecionando a partição de variáveis $\mathbf{V}$, o **Grupo de Variáveis 0 ($11$ atributos)** é formado exatamente pelos atributos corporativos da **Hewlett-Packard Labs** (onde a base Spambase foi coletada por George Forman): `word_freq_hp`, `word_freq_hpl`, `word_freq_650`, `word_freq_lab`, `word_freq_labs`, `word_freq_telnet`, `word_freq_857`, `word_freq_415`, `word_freq_85`, `word_freq_technology` e `word_freq_direct` (códigos de área 650/415 de Palo Alto, ramais 857/85 e siglas HP/HPL). Os $34$ e-mails do Cluster 1 são mensagens internas curtas da HP Labs nas quais esses 11 termos ocorrem simultaneamente com frequência extremamente alta (média de **$+6{,}86$ desvios-padrão** acima da média global).
3. **Conclusão para a Questão 1:**
   Em espaços padronizados com variáveis de cauda pesada, o Double K-Means reduz drasticamente a soma de quadrados $W$ ao criar um bloco dedicado para o subconjunto de amostras e variáveis que co-ocorrem com valores extremos ($+6{,}86\sigma$). Geometricamente, esses 34 pontos estão muito distantes da nuvem principal (daí a alta Silhueta de $0{,}6596$), mas essa separação reflete um subgrupo específico de e-mails institucionais legítimos da HP Labs, e não a fronteira global entre *spam* e *não-spam*.

---

## 4. Matriz de Protótipos dos Blocos ($\mathbf{G}$) e Partição de Variáveis

### Distribuição dos Grupos de Variáveis ($H^* = 2$)

| variable_cluster | n_features | Atributos correspondentes |
| --- | --- | --- |
| 0 | 11 | `hp`, `hpl`, `650`, `lab`, `labs`, `telnet`, `857`, `415`, `85`, `technology`, `direct` (índices 24, 25, 27–31, 33–35, 39) |
| 1 | 46 | Demais 37 palavras, 6 caracteres especiais (`;`, `(`, `[`, `!`, `\$`, `#`) e 3 métricas de maiúsculas (`capital_run_length_*`) |

### Matriz de Protótipos $\mathbf{G}_{2 \times 2}$ (na escala padronizada *z-score*)

| object_cluster | variable_cluster_0 | variable_cluster_1 |
| --- | --- | --- |
| 0 | -0.05107955 | 0.00107505 |
| 1 | 6.86118487 | -0.14440471 |

- **Interpretação da matriz $\mathbf{G}$:**
  - O bloco $(0, 0)$ ($g_{00} = -0{,}0511$) e o bloco $(0, 1)$ ($g_{01} = +0{,}0011$) representam a massa principal de $4.567$ e-mails, cujas médias padronizadas estão próximas da origem ($0$) em ambos os grupos de variáveis.
  - O bloco $(1, 0)$ ($g_{10} = +6{,}8612$) destaca o forte co-cluster formado pelos $34$ e-mails do Cluster 1 nos $11$ atributos técnicos/telefônicos da HP Labs, enquanto nas outras $46$ variáveis — bloco $(1, 1)$ ($g_{11} = -0{,}1444$) — esses mesmos e-mails apresentam frequências ligeiramente abaixo da média (por não conterem termos comerciais típicos de spam).

![Prototipos](../plots/prototypes_heatmap.png)

---

## 5. Convergência da Função Objetivo vs. Iterações

- **Execução vencedora ($(K^*, H^*) = (2, 2)$, semente `0`):**
  - **Valor inicial (iteração 0):** $W_0 = 262.251{,}38234877$
  - **Valor final (iteração 31):** $W^* = 244.486{,}69834951$ (redução de $17.764{,}68$ na soma de erros quadráticos intra-bloco)
  - **Número de iterações até convergência:** $31$ iterações (`converged = True`).
- A curva abaixo exibe o decrescimento estritamente monotônico de $W(\mathbf{G}, \mathbf{U}, \mathbf{V})$ desde a inicialização até a estabilização completa dos blocos na iteração 31:

![Convergencia](../plots/objective_convergence.png)

---

## 6. Artefatos de Reprodutibilidade e Integração com a Questão 2

Todos os resultados numéricos e partições da Questão 1 encontram-se salvos e validados em:
- [Métricas consolidadas (`q1_evaluation.json`)](../results/q1_evaluation.json)
- [Tabela de Silhueta e função objetivo (`silhouette_scores.csv`)](../results/silhouette_scores.csv)
- [Matriz de contingência (`q1_contingency.csv`)](../results/q1_contingency.csv)
- [Partição selecionada completa (`q1_selected_partition.npz`)](../results/q1_selected_partition.npz)
- [Rótulos dos objetos por linha (`q1_object_labels.csv`)](../results/q1_object_labels.csv)

Na **Questão 2**, a segunda versão do dataset (`kstar_classes`) utiliza exatamente os rótulos de objetos (`cluster` $\in \{0, 1\}$) desta partição vencedora ($4.567$ exemplos na classe `0` e $34$ exemplos na classe `1`), mantendo a correspondência linha a linha com a base Spambase.
