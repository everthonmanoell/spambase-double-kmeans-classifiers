# Status do Projeto AM 2026-2 — Spambase Double K-Means Classifiers

**Atualizado em:** 04 de novembro de 2026  
**Prazo do projeto:** 18 de novembro de 2026

Este status reflete os artefatos presentes no repositorio nesta data. Resultados
gerados localmente precisam ser transferidos junto com o codigo para estarem
disponiveis em outro computador.

## Concluido

### Questao 1 — Double K-Means

- Implementacao do algoritmo, execucao de 100 inicializacoes em cada um dos
  nove pares `(K, H)` e analise de silhueta concluidas.
- A analise registrada seleciona `K*=2`, `H*=2`, com silhueta `0.65963094`;
  convergiram 900 de 900 ajustes.
- ARI contra as classes originais: `-0.0049496989`. Os clusters nao recuperam
  a particao spam/nao spam e nao devem ser interpretados como esses rotulos.
- Resultados, graficos e interpretacao estao em
  [analysis/Q1_ANALYSIS.md](analysis/Q1_ANALYSIS.md), `results/` e `plots/`.

### Questao 2 — Classificacao

- Integracao de `kstar_classes` com os rotulos da particao real da Q1 concluida.
- Pipeline 30×10 executado e validado com MajorityVoting composto pelos quatro
  membros ajustados na validacao cruzada interna de 5-folds.
- GaussianBayes usa a covariancia de maxima verossimilhanca (dividindo por
  $N_i$), sem busca de hiperparametros (`get_param_grid() == {}`), conforme o
  PDF, com estabilizacao numerica diagonal fixa `reg_covar=1e-6` na fatoracao
  de Cholesky.
- O arquivo [results/q2/cv_30x10_results.csv](results/q2/cv_30x10_results.csv)
  contem os 3.000 resultados finais da validacao cruzada aninhada.
- A analise estatistica da tarefa 2.3 (IC 95%, Friedman, Nemenyi e boxplots)
  foi regenerada com os resultados finais e documentada em
  [analysis/Q2_STATISTICAL_ANALYSIS.md](analysis/Q2_STATISTICAL_ANALYSIS.md).
- As curvas de aprendizagem da tarefa 2.4 (5% a 95% com passo de 5%, 1.900
  avaliacoes) foram executadas e analisadas em
  [analysis/Q2_LEARNING_CURVES.md](analysis/Q2_LEARNING_CURVES.md).
- O relatorio tecnico consolidado (Q1 + Q2) foi redigido em
  [REPORT.md](REPORT.md).

## Em andamento / proximo

### Relatorio e entrega

- [x] Consolidar metodologia, resultados e conclusoes no relatorio tecnico ([REPORT.md](REPORT.md)).
- [ ] Exportar [REPORT.md](REPORT.md) para `REPORT.pdf` (se desejado) e preparar slides de ate 10 minutos.
- [ ] Organizar e transferir os arquivos finais para entrega no Google Classroom.
- [ ] Ensaiar a apresentacao da equipe.

## Saidas da tarefa 2.3

Os arquivos atualmente presentes sao preliminares e nao devem ser citados como
resultado final ate que Q2.2 seja reexecutada com MajorityVoting corrigido.

Ao executar a analise, os seguintes arquivos sao criados ou atualizados:

- `results/confidence_intervals.csv`
- `results/friedman_results.txt`
- `results/nemenyi_pvalues.csv`
- `plots/comparison_boxplots.png`

Os intervalos de confianca usam as medias de cada uma das 30 repeticoes (dez
folds por repeticao), evitando considerar os 300 folds sobrepostos como
observacoes independentes para esse calculo. O teste de Friedman usa os 300
folds alinhados entre classificadores; como os folds se sobrepoem, interpretar
seus p-valores com cautela.

## Comandos uteis

```bash
# Rodar a analise estatistica sobre o CSV ja gerado
python -m src.analysis.statistical_analysis

# Executar testes
pytest tests/ -v
```

## Observacao sobre os documentos de planejamento

O [ACTION_PLAN.md](ACTION_PLAN.md) ainda contem marcacoes anteriores a
execucao da Q1 e das tarefas 2.1-2.3. O [CHECKLIST.md](CHECKLIST.md) foi
atualizado em 03/11: a implementacao da tarefa 2.4 esta pronta, mas a execucao
real continua pendente.
