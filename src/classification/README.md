## Decisões de Arquitetura: Classificador Bayesiano Baseado em k-Vizinhos

### 1. Justificativa Teórica: A Equivalência Bayesiana
O projeto exige a implementação de um classificador bayesiano utilizando a estimativa de densidade por k-vizinhos e a estimativa de máxima verossimilhança para a probabilidade a priori das classes $P(\omega_i)$. Matematicamente, a regra de classificação padrão do algoritmo k-NN é idêntica à formulação bayesiana. 

Na estimativa de densidade baseada em vizinhança, a função de verossimilhança é dada por:
$$p(x \vert{} \omega_i) \approx \frac{k_i}{N_i V}$$
Onde $k_i$ é o número de vizinhos da classe $\omega_i$ dentro da vizinhança de volume $V$, e $N_i$ é o total de exemplos da classe. Utilizando a estimativa de máxima verossimilhança para a priori ($P(\omega_i) = \frac{N_i}{N}$), e aplicando o Teorema de Bayes para a probabilidade a posteriori:
$$P(\omega_i \vert{} x) = \frac{p(x \vert{} \omega_i) P(\omega_i)}{p(x)} = \frac{\frac{k_i}{N_i V} \frac{N_i}{N}}{\frac{k}{N V}} = \frac{k_i}{k}$$
Como resultado, a alocação da amostra à classe com a maior densidade a posteriori equivale diretamente a escolher a classe majoritária entre os $k$ vizinhos. Por esta razão, a classe nativa do Scikit-Learn foi envelopada, pois ela já processa eficientemente a razão $\frac{k_i}{k}$.

### 2. Definição da Grade de Busca (Hyperparameter Grid)
* **Métrica de Distância (`metric`):** Em estrito atendimento ao documento do projeto, a grade de busca fixará a vizinhança testando as distâncias Euclidiana, City-Block (conhecida como Manhattan na biblioteca) e Chebishev.
* **Número de Vizinhos (`n_neighbors`):** Optou-se por uma grade de valores ímpares (`[1, 3, 5, 7, 11, 15, 21, 31]`) para prevenir empates em deliberações majoritárias bidimensionais. A progressão não linear permite ao `GridSearchCV` investigar o comportamento do modelo desde contornos de decisão altamente ruidosos ($k=1$) até fronteiras mais suaves e globalizadas ($k=31$), sem inflar o custo da validação cruzada aninhada $30 \times 10$-folds.