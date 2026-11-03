# spambase-double-kmeans-classifiers

## Questao 1

Com as dependencias do projeto instaladas, execute na raiz:

```bash
python -m src.experiments.question_1 --n-runs 2 --output-dir results/q1_smoke
python -m src.experiments.question_1
```

O primeiro comando valida o fluxo com 18 ajustes; o segundo executa os 900
ajustes previstos no plano, usando dados reais padronizados, sem mock.
Com Poetry, use `poetry run python` no lugar de `python`.

Veja [protocolo, retomada e arquivos de saida](src/experiments/README.md).

Depois dos 900 ajustes, calcule a silhueta dos nove vencedores e selecione
`(K*, H*)`, sem refazer o clustering:

```bash
~/.local/bin/poetry run python -m src.analysis.silhouette
```

O comando salva os scores e a selecao em `results/`, e o heatmap em `plots/`.

Para concluir a Q1 com ARI, contingencia, prototipos, convergencia e relatorio:

```bash
~/.local/bin/poetry run python -m src.analysis.clustering_evaluation
```

Esse comando atualiza a silhueta, sem refazer o clustering, e gera
[a analise da Q1](analysis/Q1_ANALYSIS.md).