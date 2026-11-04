# ✅ CHECKLIST EXECUTIVO - Projeto AM 2026-2

**Deadline:** 18 de Novembro, 2026 (16 dias)  
**Atualizado:** 03 de Novembro, 2026

---

## 📊 QUESTÃO 1: Double K-Means (Prazo: 08 Nov)

- [x] **1.1 - Script Execução em Larga Escala** (4-6 horas)
  - [x] Criar arquivo `src/experiments/question_1.py`
  - [x] Função para rodar Double K-Means 100 vezes para cada (K,H)
  - [x] K ∈ {2,3,4}, H ∈ {1..K}
  - [x] Salvar melhores resultados em `results/q1_executions_summary.csv`
  - [x] Salvar histórico completo em `results/q1_full_history.pkl`
  - [x] Teste: confirmar que tem 9 pares (K,H) × 100 rodadas = 900 execuções
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

- [x] **1.2 - Análise de Silhueta** (2-3 horas)
  - [x] Criar arquivo `src/analysis/silhouette.py`
  - [x] Calcular `silhouette_score()` para cada (K,H)
  - [x] Encontrar K* = argmax(Sil)
  - [x] Salvar scores em `results/silhouette_scores.csv`
  - [x] Gerar plot `plots/silhouette_heatmap.png`
  - [x] Testar: confirmar que K* foi encontrado
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

- [x] **1.3 - Matriz de Confusão & ARI** (2-3 horas)
  - [x] Criar arquivo `src/analysis/clustering_evaluation.py`
  - [x] Carregar melhor resultado com K*
  - [x] Calcular `adjusted_rand_index()` contra labels originais
  - [x] Gerar matriz de confusão
  - [x] Salvar em `results/ari_score.txt`
  - [x] Gerar plot `plots/confusion_matrix_q1.png`
  - [x] Comentários em `analysis/Q1_ANALYSIS.md`
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

- [x] **1.4 - Visualizações** (2 horas)
  - [x] Criar arquivo `src/visualization/q1_plots.py`
  - [x] Plot 1: Matriz de protótipos (G) → `plots/prototypes_heatmap.png`
  - [x] Plot 2: Função objetivo vs iterações → `plots/objective_convergence.png`
  - [x] Plot 3: Silhueta heatmap (já em 1.2)
  - [x] Todos os plots com títulos, labels, legendas
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

**Q1 STATUS:** ████████████████ 100%
**Q1 DEADLINE:** Sexta, 08 de Novembro

---

## 📊 QUESTÃO 2: Classificação (Prazo: 15 Nov)

### Phase 2.A: Integração (1-2 horas)

- [x] **2.1 - K* com Dataset** (1-2 horas)
  - [x] Modificar `src/utils/dataloader.py`
  - [x] Função `load_spambase_kstar_real()` que usa K* real
  - [x] Testar que carrega corretamente com K* classes
  - [x] Print: K* value e distribuição de classes
  - **Status:** Feito ✅ — usa os rótulos reais da partição da Q1; não são as classes spam/não spam.
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

### Phase 2.B: Pipeline Experimental (6-8 horas)

- [x] **2.2 - CV Estratificada 30×10** (6-8 horas)
  - [x] Criar arquivo `src/experiments/question_2.py`
  - [x] Implementar CV aninhada:
    - [x] Outer: RepeatedStratifiedKFold(n_splits=10, n_repeats=30)
    - [x] Inner: StratifiedKFold(n_splits=5) para tuning
  - [x] Para cada fold externo:
    - [x] Fazer grid search com CV interna
    - [x] Treinar com melhores hiperparâmetros
    - [x] Avaliar no fold teste
  - [x] Rodar para 2 datasets × 5 classifiers
  - [x] Reexecutar com MajorityVoting composto pelos quatro estimadores tunados
  - [x] Tratar covariância singular no GaussianBayes usando estimativa MV + estabilização numérica fixa (`reg_covar=1e-6`, sem GridSearch)
  - [x] Salvar resultados corrigidos em `results/q2/cv_30x10_results.csv`
  - [x] Incluir colunas: fold_id, classifier, dataset, error, precision, recall, f1, hyperparams
  - [x] Testar: confirmar 30×10×2×5 = 3000 linhas de resultados
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** 04 Nov 2026

### Phase 2.C: Análise Estatística (3-4 horas)

- [x] **2.3 - Testes Estatísticos** (3-4 horas)
  - [x] Criar arquivo `src/analysis/statistical_analysis.py`
  - [x] Completar `perform_friedman_nemenyi()` em `src/utils/statistical_tests.py`
  - [x] Para cada métrica (error, precision, recall, f1):
    - [x] Calcular média e IC 95% para cada classificador usando as médias das 30 repetições
    - [x] Executar Friedman test para cada dataset
    - [x] Se p < 0.05: executar Nemenyi post-hoc
  - [x] Salvar IC em `results/confidence_intervals.csv`
  - [x] Salvar Friedman em `results/friedman_results.txt`
  - [x] Salvar Nemenyi em `results/nemenyi_pvalues.csv`
  - [x] Gerar e revisar o plot `plots/comparison_boxplots.png`
  - [x] Documentar metodologia, resultados e limitações em `analysis/Q2_STATISTICAL_ANALYSIS.md`
  - [x] Imprimir métricas, ICs e testes no terminal
  - [x] Reexecutar após a Q2.2 corrigida
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** 04 Nov 2026

### Phase 2.D: Curva de Aprendizagem (3-4 horas)

- [x] **2.4 - Learning Curves** (3-4 horas)
  - [x] Criar arquivo `src/experiments/learning_curves.py`
  - [x] Implementar treino/teste para ambos datasets e cinco classificadores
  - [x] Implementar tamanhos de treino 5%, 10%, ..., 95%, com dez divisões estratificadas
  - [x] Selecionar hiperparâmetros dos quatro modelos base por CV interna estratificada (F1 macro); Voting usa os estimadores selecionados
  - [x] Salvar hiperparâmetros escolhidos e número de folds internos por split
  - [x] Calcular F1 macro no treino e no teste
  - [x] Implementar gráficos com média e faixa de ±1 desvio-padrão
  - [x] Implementar salvamento dos dados detalhados e resumo
  - [x] Executar nos dados reais e verificar as 1.900 avaliações
  - [x] Confirmar os gráficos `plots/learning_curves_original.png` e `plots/learning_curves_kstar.png`
  - [x] Registrar observações de sobreajuste/subajuste a partir dos resultados reais
  - [x] Documentar protocolo em `analysis/Q2_LEARNING_CURVES.md`
  - **Status:** Feito ✅
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** 04 Nov 2026

**Q2 STATUS:** ████████████████ 100%
**Q2 DEADLINE:** Sábado, 15 de Novembro
**Nota:** Todas as tarefas da Questão 2 (2.1, 2.2, 2.3 e 2.4) estão 100% executadas e documentadas.

---

## 📊 QUESTÃO 3: Testes Adicionais

- [x] **Implementação dos 5 Classificadores** ✅ COMPLETO
- [x] **Validar em Pipeline Q2**
- [ ] **Confirmar conformidade com especificações**

**Q3 STATUS:** ██████████░░░░░░ 67%

---

## 📋 DOCUMENTAÇÃO & ENTREGA (Prazo: 18 Nov)

- [x] **3.1 - Relatório Técnico** (4-6 horas)
  - [x] Criar arquivo `REPORT.pdf` ou `REPORT.md`
  - [x] Seções:
    - [x] Capa (título, autores, data, instituição)
    - [x] Resumo executivo (1 página)
    - [x] Introdução (contexto, objetivo, motivação)
    - [x] Descrição dos dados (spambase)
    - [x] Metodologia (Q1, Q2, Q3)
    - [x] Resultados (tabelas, gráficos, testes estatísticos)
    - [x] Discussão (interpretação dos resultados)
    - [x] Conclusão
    - [x] Referências
  - **Status:** ✅ Feito em `REPORT.md`
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

- [ ] **3.2 - Slides de Apresentação** (3-4 horas)
  - [ ] Criar arquivo `PRESENTATION.pptx` ou `PRESENTATION.pdf`
  - [ ] ~15-20 slides
  - [ ] Conteúdo:
    - [ ] Capa
    - [ ] Objetivo
    - [ ] Dataset
    - [ ] Metodologia Q1
    - [ ] Resultados Q1
    - [ ] Metodologia Q2
    - [ ] Resultados Q2 (tabelas)
    - [ ] Resultados Q2 (gráficos)
    - [ ] Testes Estatísticos
    - [ ] Curva de Aprendizagem
    - [ ] Conclusões
  - [ ] Todos os slides legíveis
  - [ ] Máximo 10 minutos (testar)
  - **Status:** ❌ Não iniciado
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

- [ ] **3.3 - Preparação Entrega Google Classroom** (1 hora)
  - [ ] Criar arquivo `README.md` com instruções
  - [ ] Verificar estrutura de diretórios
  - [ ] Preparar arquivos individuais (NÃO .zip):
    - [ ] Código fonte: todos os `.py` em `src/`
    - [ ] Testes: todos os `.py` em `tests/`
    - [ ] Relatório: `REPORT.pdf`
    - [ ] Slides: `PRESENTATION.pptx`
    - [ ] Resultados principais: `results/` (CSVs com números)
  - [ ] Upload no Google Classroom
  - **Status:** ❌ Não iniciado
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

- [ ] **3.4 - Preparação para Apresentação** (2-3 horas)
  - [ ] Todos os membros preparados
  - [ ] Divisão de tópicos entre apresentadores
  - [ ] Ensaio completo da apresentação (cronometrar 10 min)
  - [ ] Preparar para possíveis perguntas
  - [ ] Confirmar dia/hora da apresentação
  - **Status:** ❌ Não iniciado
  - **Responsável:** ?
  - **Data Início:** ?
  - **Data Fim:** ?

**DOCS STATUS:** ░░░░░░░░░░░░░░░░ 0%
**DOCS DEADLINE:** Segunda, 18 de Novembro

---

## 📅 CRONOGRAMA SEMANAL

### SEMANA 1: 02-08 de Novembro (14 horas + computação)
| Data | Tarefa | Horas | Responsável |
|------|--------|-------|-------------|
| 02-03 | 1.1: Script execução | 6 | |
| 04 | 1.2: Silhueta | 3 | |
| 05 | 1.3: ARI + Confusão | 3 | |
| 06-07 | 1.4: Plots | 2 | |
| **08** | **✅ Q1 PRONTO** | | |

### SEMANA 2: 09-15 de Novembro (18 horas + computação)
*Cronograma planejado originalmente; as tarefas 2.1-2.3 já foram concluídas antes dessas datas.*
| Data | Tarefa | Horas | Responsável |
|------|--------|-------|-------------|
| 09 | 2.1: K* integração | 2 | |
| 10-11 | 2.2: Pipeline CV | 8 | |
| 12 | 2.3: Testes Friedman | 4 | |
| 13-14 | 2.4: Curva aprendizagem | 4 | |
| **15** | **✅ Q2 PRONTO** | | |

### SEMANA 3: 16-18 de Novembro (9 horas)
| Data | Tarefa | Horas | Responsável |
|------|--------|-------|-------------|
| 16-17 | 3.1 + 3.2: Relatório + Slides | 8 | |
| **18** | 3.3 + 3.4: Entrega + Apresentação | 1 | |

**TOTAL: ~41 horas de trabalho**

---

## ✨ PRÓXIMAS AÇÕES (03 Nov)

- [ ] Consolidar resultados Q1/Q2 no relatório técnico e preparar slides.
- [ ] Revisar conformidade dos classificadores com as especificações do projeto.
- [ ] Executar o script de curvas de aprendizagem e documentar os resultados.
- [ ] Reexecutar Q2.2 com os modelos base tunados para MajorityVoting; regenerar Q2.3 em seguida.

---

## 📝 NOTAS IMPORTANTES

1. **Dependências críticas:**
   - Q1.2 (Silhueta) depende de Q1.1 (execução)
   - Q2.1 (K* integração) depende de Q1.2 (encontrar K*)
   - Q2.2 (Pipeline) depende de Q2.1 (dataset pronto)
   - Documentação depende de todos os resultados

2. **Arquivos de saída esperados:**
   - `results/`: CSV com dados, TXT com análises
   - `plots/`: PNG com gráficos
   - Root: REPORT.pdf, PRESENTATION.pptx, README.md

3. **Formato de entrega:**
   - ⚠️ Arquivos INDIVIDUAIS, NÃO em .zip
   - Todos os `.py` files separados
   - Cada arquivo PDF/PPTX separado

4. **Qualidade esperada:**
   - Todos os scripts com docstrings
   - Plots com títulos e labels
   - Código segue PEP 8
   - Relatório tem todas as 9 seções

---

## 🎯 MARCOS IMPORTANTES

✅ **02 Nov** - Análise do projeto (FEITO!)
⏳ **08 Nov** - Deadline Q1 (Questão 1 completa)
⏳ **15 Nov** - Deadline Q2 (concluir curva e revisão)
🎯 **18 Nov** - Entrega final + Apresentação oral

---

**Última Atualização:** 03 de Novembro, 2026 - 21:58
