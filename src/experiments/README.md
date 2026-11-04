# Execucao da Questao 1

O experimento usa `load_spambase_original`, sem rotulos mock. Aplica
`StandardScaler` (z-score por atributo) sobre os 57 atributos, conforme a
decisao documentada em [clustering](../clustering/README.md). A classe original
nao entra no ajuste. Nenhuma linha e reordenada ou removida.

Na raiz do projeto, com as dependencias de `pyproject.toml` instaladas:

```bash
# Validacao reduzida: 2 inicializacoes por par, 18 ajustes sobre os dados reais.
python -m src.experiments.question_1 --n-runs 2 --output-dir results/q1_smoke

# Experimento completo: 100 inicializacoes por par, 900 ajustes.
python -m src.experiments.question_1

# Retomar com exatamente os mesmos dados, n-runs, max-iter e versoes das bibliotecas.
python -m src.experiments.question_1 --resume
```

Se estiver usando Poetry, prefixe os comandos com `poetry run`.
`--data` permite informar outro caminho para o CSV sem cabecalho.
`--max-iter` controla o limite de seguranca do algoritmo (padrao: 100).
O comando deve ser executado por um unico processo por diretorio de saida.

## Protocolo

- Grade fixa: `K` em `{2, 3, 4}`, `H` de `1` ate `K` (9 pares).
- Sementes: `random_state=run`, de `0` a `n_runs - 1`, em cada par.
- Vencedor de cada par: menor funcao objetivo; em empate exato, menor `run`.
- Execucoes sem convergencia sao mantidas, com aviso explicito. Consulte
  `convergence_rate` e `best_converged` antes de interpretar os resultados.
- O checkpoint e salvo atomicamente ao terminar cada par. Uma interrupcao
  durante um par exige repetir somente esse par; os anteriores sao reutilizados.
- Sem `--resume`, um checkpoint existente causa erro, evitando sobrescrita
  acidental. Para mudar parametros, use outro `--output-dir`.

## Arquivos de saida

- `q1_executions_summary.csv`: uma linha por par concluido, com `K`, `H`,
  `n_runs`, `best_objective`, `best_run_id`, `n_iterations`, `convergence_rate`
  (fracao de 0 a 1) e `best_converged`. O experimento completo tem 9 linhas.
- `q1_full_history.pkl`: dicionario com `metadata`, `scaler_mean`,
  `scaler_scale` e `runs`. O historico completo tem 900 registros, cada um com
  `run` (tambem a semente), `K`, `H`, `objective`, `n_iter`, `converged`,
  `n_relocations`, `labels`, `column_labels`, `prototypes` e `objective_history`.
  Os metadados incluem hash SHA-256 do CSV e configuracao para validar a retomada.
- `q1_best_dkm_K{K}_H{H}.npz`: os mesmos campos do registro vencedor de cada
  par, carregaveis com `numpy.load(..., allow_pickle=False)`.

O historico e a fonte de verdade da retomada; CSV e arquivos NPZ sao
reconstruidos a partir dele. Carregue pickle apenas de origem local confiavel.
Os parametros do scaler permitem reconstruir a matriz padronizada para a
analise de silhueta e os graficos.

## Analise de silhueta (Tarefa 1.2)

Depois de concluir os nove pares, execute:

```bash
~/.local/bin/poetry run python -m src.analysis.silhouette
```

O comando nao repete o clustering. Usa o historico completo como fonte de
verdade e recupera, por par, a execucao de menor objetivo. Reconstroi a mesma
matriz padronizada usando `scaler_mean` e `scaler_scale` salvos e calcula a
silhueta euclidiana completa, sem amostragem, sem usar a classe original.
Os dados (hash e ordem), a grade e as versoes NumPy, pandas e scikit-learn
devem corresponder ao experimento. Checkpoints incompletos sao rejeitados.
Use apenas historicos pickle locais confiaveis.

O maior score determina `(K*, H*)`. Em empate exato, vence o menor `K` e depois
o menor `H`. As combinacoes `H > K` ficam vazias no heatmap, nao recebem zero.
Execucoes vencedoras sem convergencia sao mantidas com aviso explicito,
seguindo o protocolo de selecao da Tarefa 1.1.

Saidas:

- `results/silhouette_scores.csv`: nove linhas, com `K`, `H`, `n_runs`,
  `best_run_id`, `best_objective`, `best_converged` e `silhouette`.
- `results/q1_selection.json`: o registro vencedor com os mesmos campos, mais
  metrica, padronizacao e hash do dataset. Identifica `(K*, H*, best_run_id)`
  para recuperar os rotulos no historico, sem depender de um novo ajuste.
- `plots/silhouette_heatmap.png`: comparacao dos nove scores na grade `K x H`.

Os caminhos podem ser alterados com `--data`, `--results-dir` e `--plots-dir`.
Para analisar uma execucao reduzida, informe seu diretorio de resultados;
o comando emitira um aviso de que nao corresponde as 100 inicializacoes.
Reexecutar a analise atualiza essas tres saidas sem alterar o historico da Q1.

## Avaliacao e relatorio final da Q1 (Tarefas 1.3 e 1.4)

```bash
~/.local/bin/poetry run python -m src.analysis.clustering_evaluation
```

O comando atualiza a silhueta e a selecao a partir do historico completo,
sem repetir nenhum ajuste, e compara a particao selecionada com as classes
originais. O ARI e invariante a permutacoes dos IDs dos clusters.
A matriz de confusao e apresentada como tabela de contingencia retangular
(classes nas linhas, clusters nas colunas), sem associar IDs de clusters a
classes spam/nao spam nem tratar a diagonal como acuracia.

Saidas adicionais:

- `results/q1_evaluation.json`: metricas, selecao, convergencia e hash dos dados.
- `results/ari_score.txt`: ARI com interpretacao.
- `results/q1_contingency.csv`: contagens classe original x cluster.
- `results/q1_cluster_sizes.csv` e `q1_variable_cluster_sizes.csv`: tamanhos.
- `results/q1_prototypes.csv`: G na escala padronizada.
- `results/q1_objective_history.csv`: W na inicializacao (0) e por iteracao.
- `results/q1_object_labels.csv`: indice original da linha (base zero),
  classe original e cluster.
- `results/q1_variable_labels.csv`: indice do atributo (base zero) e cluster.
- `results/q1_selected_partition.npz`: todos os campos do ajuste vencedor,
  parametros do scaler e hash do dataset, sem objetos pickle.
- `plots/confusion_matrix_q1.png`, `prototypes_heatmap.png` e
  `objective_convergence.png`: graficos exigidos.
- `analysis/Q1_ANALYSIS.md`: protocolo, dados, resultados e interpretacao.

Os caminhos sao configuraveis por `--data`, `--results-dir`, `--plots-dir`
e `--report-path`. Reexecutar atualiza as saidas, mas preserva o historico.
O comando tambem aceita experimentos reduzidos para testes, com aviso explicito
no log e no relatorio: a entrega exige 100 inicializacoes por par.

Guarde backup dos resultados e dos graficos: estao ignorados pelo Git.
O relatorio Markdown pode ser versionado, mas suas imagens e tabelas externas
precisam ser preservadas junto dele.
Os rotulos do vencedor de `(K*, H*)`, na ordem original das linhas,
serao usados na Q2; nao se deve gerar classes aleatorias nem refazer um ajuste
independente para substituir esses rotulos.
