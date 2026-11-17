# Verificação das referências do relatório

**Data:** 2026-11-11
**Arquivos verificados:** `docs/relatorio/referencias.bib` (27 entradas, biblatex/biber, `style=ieee`) e `docs/relatorio/relatorio.tex` (citações nas linhas 225–1247).

## Resumo

Foram verificadas as 27 entradas de `referencias.bib`. Todas as 27 obras **existem** e foram localizadas em fonte primária ou equivalente: Crossref/DataCite para os DOIs, páginas do JMLR e do IJCAI, catálogo da Library of Congress e catálogo da biblioteca de Princeton. **Nenhuma referência deixou de ser encontrada.** **22 entradas estão OK**, sem erro de metadados que precise de correção (algumas têm melhorias opcionais, listadas no fim). **3 entradas precisam de correção**: `vichi2001double` (falta `series`), `govaert2013coclustering` (falta o subtítulo, editora pouco precisa) e `kohavi1995study` (páginas erradas). **2 entradas ficaram em "atenção"** porque não são citadas no texto e, portanto, o biblatex não as imprime: `pedregosa2011scikit` e também `bouckaert2004evaluating`, que o pedido não mencionava. Os metadados dessas duas estão corretos. Todas as 25 chaves citadas existem no `.bib`: o `relatorio.blg` registra 25 citekeys sem avisos, e o `relatorio.log` não tem citações indefinidas. No uso das citações, há 4 pontos de atenção (linhas 353, 688, 702 e 1191) e 2 sugestões de citação que aproveitariam as entradas não usadas (linhas 646/702 e 1197/1247).

> Método: cada DOI foi consultado em `https://api.crossref.org/works/<DOI>`, ou na DataCite para o DOI da UCI, sem e-mail no User-Agent. Os livros sem DOI foram conferidos no catálogo da Library of Congress (SRU `lx2.loc.gov`). Os artigos do JMLR e do IJCAI foram conferidos pelas páginas oficiais (meta tags `citation_*`). As páginas da Springer (link.springer.com), Wiley (onlinelibrary/wiley.com), JSTOR, Project Euclid e DBLP responderam com desafio anti-robô ou erro 403. **Esses bloqueios não foram contornados.** Nesses casos, a fonte usada foi o registro Crossref depositado pela própria editora.

## Tabela resumo

| Chave | Existe? | Status | Problema (uma linha) |
|---|---|---|---|
| `hopkins1999spambase` | sim | OK | — (o registro DataCite tem criadores malformados; o `.bib` segue a página da UCI, que está correta) |
| `vichi2001double` | sim | **corrigir** | Falta `series = {Studies in Classification, Data Analysis, and Knowledge Organization}` |
| `govaert2013coclustering` | sim | **corrigir** | Título sem o subtítulo "Models, Algorithms and Applications"; `publisher` "ISTE/Wiley" → "ISTE and Wiley"; sem DOI |
| `hartigan1979kmeans` | sim | OK | — |
| `rousseeuw1987silhouettes` | sim | OK | — |
| `kaufman1990finding` | sim | OK | — (opcional: `series`, DOI) |
| `hubert1985comparing` | sim | OK | — |
| `duda2001pattern` | sim | OK | — (LoC registra a editora como "Wiley"; "Wiley-Interscience" é aceitável) |
| `hastie2009elements` | sim | OK | Metadados OK; uso fraco na l. 702 (ver "Uso das citações") |
| `cover1967nearest` | sim | OK | — |
| `parzen1962estimation` | sim | OK | Metadados OK; o uso na l. 688 atribui a Parzen o caso multivariado, que não está no artigo |
| `silverman1986density` | sim | OK | — (opcional: `series`) |
| `ledoit2004wellconditioned` | sim | OK | — |
| `kuncheva2014combining` | sim | OK | Metadados OK; uso fraco na l. 1191 |
| `kohavi1995study` | sim | **corrigir** | `pages` 1137–1145 → 1137–1144 (IJCAI); `booktitle` não segue o título oficial |
| `varma2006bias` | sim | OK | — (opcional: 91 é número de artigo, não página) |
| `cawley2010overfitting` | sim | OK | — (opcional: `number = {70}`) |
| `dietterich1998approximate` | sim | OK | — |
| `nadeau2003inference` | sim | OK | — |
| `bengio2004noUnbiased` | sim | OK | — |
| `bouckaert2004evaluating` | sim | **atenção** | Metadados corretos, mas **não é citada** no texto (não aparece na bibliografia) |
| `friedman1937ranks` | sim | OK | — |
| `nemenyi1963distribution` | sim | OK | — |
| `demsar2006statistical` | sim | OK | Metadados OK; uso na l. 353 merece ressalva |
| `benavoli2016posthoc` | sim | OK | — |
| `sokolova2009systematic` | sim | OK | — |
| `pedregosa2011scikit` | sim | **atenção** | Metadados corretos, mas **não é citada** (o scikit-learn é usado nas l. 408, 646, 673, 702 e 710 sem citação) |

Contagem: **22 OK · 3 corrigir · 2 atenção (não citadas) · 0 não encontradas.**

---

## Entradas que precisam de correção

### `vichi2001double` — corrigir (menor)

- **Conferido no Crossref** (https://api.crossref.org/works/10.1007/978-3-642-59471-7_6):
  - título "Double k-means Clustering for Simultaneous Classification of Objects and Variables" ✓
  - páginas 43–52 ✓, ano 2001 ✓
  - editora Springer Berlin Heidelberg, local "Berlin, Heidelberg" ✓
  - o DOI resolve para este capítulo ✓
- **Livro** (https://api.crossref.org/works/10.1007/978-3-642-59471-7): "Advances in Classification and Data Analysis", com editores Simone Borra, Roberto Rocci, Maurizio Vichi e Martin Schader ✓ (mesma ordem do `.bib`).
- **Problema:** o livro pertence à série **"Studies in Classification, Data Analysis, and Knowledge Organization"** (ISSN 1431-8814, `container-title` no Crossref), e o campo `series` está ausente. O `biblatex-ieee` imprime `series` em `@incollection` como "ser. …" (`ieee.bbx`, `\DeclareFieldFormat[incollection,inproceedings]{series}`).
- **Observação:** no registro Crossref do capítulo, o nome do autor vem invertido (`given = "Vichi"`, `family = "Maurizio"`). É um erro do depósito; o registro do livro traz o editor "Maurizio Vichi". O `.bib` está certo, e não se deve reimportar essa entrada do Crossref.
- A página da Springer (link.springer.com/chapter/10.1007/978-3-642-59471-7_6) exigiu desafio anti-robô e não foi consultada.

```bibtex
@incollection{vichi2001double,
  author    = {Vichi, Maurizio},
  title     = {Double k-means Clustering for Simultaneous Classification of Objects and Variables},
  booktitle = {Advances in Classification and Data Analysis},
  editor    = {Borra, Simone and Rocci, Roberto and Vichi, Maurizio and Schader, Martin},
  series    = {Studies in Classification, Data Analysis, and Knowledge Organization},
  publisher = {Springer},
  address   = {Berlin, Heidelberg},
  year      = {2001},
  pages     = {43--52},
  doi       = {10.1007/978-3-642-59471-7_6}
}
```

### `govaert2013coclustering` — corrigir

- **Conferido no Crossref** (https://api.crossref.org/works/10.1002/9781118649480):
  - título "Co‐Clustering" com **subtítulo "Models, Algorithms and Applications"**
  - autores Gérard Govaert e Mohamed Nadif ✓
  - editora "Wiley", data de impressão 2013-12-02 ✓ (online em 2014-02-10)
  - ISBN impresso 978-1-84821-473-6, eletrônico 978-1-118-64948-0
  - DOI 10.1002/9781118649480
- **Problemas:**
  1. O `title` omite o subtítulo.
  2. `publisher = {ISTE/Wiley}`: no biblatex, várias editoras se separam com `and`. A barra imprime "ISTE/Wiley" literalmente, o que funciona, mas `ISTE and Wiley` é a forma correta do campo de lista.
  3. Falta o DOI.
- **Não verificado:** o local "London" e a coedição pela ISTE. As páginas da Wiley (onlinelibrary.wiley.com e wiley.com) responderam 403, e o catálogo da LoC não tem o título. As listagens que aparecem na busca dão "ISTE e Wiley" como coeditoras (por exemplo, https://www.lehmanns.ch/shop/mathematik-informatik/25704173-9781848214736-co-clustering, que também respondeu 403 ao acesso direto). Um registro de catálogo visto apenas em resultado de busca (https://find.library.upatras.gr/Record/ocn872372294, conexão recusada) traz "London : Hoboken : ISTE ; John Wiley & Sons, 2014". Por isso, o local "London" foi mantido. O ano 2013 segue o Crossref; alguns catálogos registram 2014.

```bibtex
@book{govaert2013coclustering,
  author    = {Govaert, G{\'e}rard and Nadif, Mohamed},
  title     = {Co-Clustering: Models, Algorithms and Applications},
  publisher = {ISTE and Wiley},
  address   = {London},
  year      = {2013},
  doi       = {10.1002/9781118649480}
}
```

### `kohavi1995study` — corrigir

- **Página oficial do IJCAI** (https://www.ijcai.org/proceedings/1995/147):
  - autor Ron Kohavi ✓; título idêntico ✓; ano 1995 ✓
  - anais "Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence, IJCAI-95", editor Chris S. Mellish, editora "International Joint Conferences on Artificial Intelligence"
  - **páginas 1137–1144** (meta tags `citation_firstpage`/`citation_lastpage` e BibTeX da própria página)
  - o índice do volume (https://www.ijcai.org/Proceedings/95-2/) também lista "Pages 1137–1144"; o artigo seguinte começa na p. 1146
- No PDF oficial (https://ijcai.org/proceedings/1995/0147.pdf), as páginas numeradas vão de 1137 a 1143; a p. 1144 provavelmente está em branco. **Nenhuma fonte do IJCAI apoia 1137–1145.**
- **Problemas:** `pages = {1137--1145}` → `1137--1144`. O `booktitle` atual ("Proceedings of the 14th … (IJCAI)") é aceitável, mas não é o título oficial. O volume dos anais (o caminho da URL sugere vol. 2) **não foi verificado**, porque a página atual do IJCAI não indica volumes; por isso, ele não entra na entrada corrigida.

```bibtex
@inproceedings{kohavi1995study,
  author    = {Kohavi, Ron},
  title     = {A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection},
  booktitle = {Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence (IJCAI-95)},
  editor    = {Mellish, Chris S.},
  publisher = {International Joint Conferences on Artificial Intelligence},
  pages     = {1137--1144},
  year      = {1995},
  url       = {https://www.ijcai.org/proceedings/1995/147}
}
```

---

## Entradas não citadas (atenção)

Contagem de ocorrências de cada chave em `relatorio.tex` (grep): `bouckaert2004evaluating` = 0 e `pedregosa2011scikit` = 0. Não existe `\nocite` no documento. O `relatorio.bbl` contém 25 entradas, sem essas duas. Com `biblatex`, elas **não aparecem na lista de Referências**.

### `pedregosa2011scikit` — atenção (metadados corretos)

- **JMLR** (https://www.jmlr.org/papers/v12/pedregosa11a.html):
  - título "Scikit-learn: Machine Learning in Python" ✓
  - os 16 autores conferem na mesma ordem e grafia (Gaël Varoquaux, Édouard Duchesnay) ✓
  - JMLR 12(85):2825–2830, 2011 ✓
- Opcional: `number = {85}`.
- Com `maxbibnames=6`, a entrada sairia como "F. Pedregosa *et al.*", o que é normal no estilo IEEE.
- **Sugestão:** citar na l. 646 ("Todos os classificadores seguem a interface do scikit-learn") e/ou na l. 702 (ver abaixo). Assim a entrada é aproveitada em vez de removida.

### `bouckaert2004evaluating` — atenção (metadados corretos)

- **Crossref** (https://api.crossref.org/works/10.1007/978-3-540-24775-3_3):
  - autores Remco R. Bouckaert e Eibe Frank ✓; título ✓; páginas 3–12 ✓; 2004 ✓
  - série Lecture Notes in Computer Science ✓; livro "Advances in Knowledge Discovery and Data Mining" ✓
  - editores Honghua Dai, Ramakrishnan Srikant e Chengqi Zhang (https://api.crossref.org/works/10.1007/b97861)
- **Volume 3056:** não aparece no registro Crossref. Ele aparece na página da Springer do livro (https://link.springer.com/book/10.1007%2Fb97861), vista apenas por resultado de busca porque o acesso direto tem desafio anti-robô. O repositório da Universidade de Waikato (https://researchcommons.waikato.ac.nz/handle/10289/1451) confirma o evento (PAKDD 2004, Sydney), as páginas 3–12 e o DOI.
- **Sugestão:** citar junto com Nadeau e Bengio nas l. 1197 e 1247. Demšar (2006, §2) relata que Bouckaert e Frank preferiram o teste *t* com reamostragem corrigida ao 5×2cv, que é justamente a alternativa mencionada ali para a validação cruzada repetida 30×10. Se a entrada não for citada, pode ser removida do `.bib`.

---

## Entradas OK (com fonte)

- **`hopkins1999spambase`**
  - A página da UCI (https://archive.ics.uci.edu/dataset/94/spambase) lista os criadores Mark Hopkins, Erik Reeber, George Forman e Jaap Suermondt (mesma ordem do `.bib`), com doação em 1999-06-30 e DOI 10.24432/C53G6X. A citação sugerida pela própria UCI coincide com a entrada.
  - O DOI é da **DataCite**, não do Crossref: a API do Crossref responde "Resource not found". Na DataCite (https://api.datacite.org/dois/10.24432/C53G6X): título "Spambase", publisher "UCI Machine Learning Repository", 1999, tipo Dataset.
  - O campo `creators` da DataCite está malformado: um único criador, "Mark Hopkins, Erik Reeber". **Não importar daí.**
  - `@misc` com `howpublished` é adequado; `@dataset` também existiria no biblatex padrão, mas não é necessário.
- **`hartigan1979kmeans`**
  - No Crossref (https://api.crossref.org/works/10.2307/2346830): "Algorithm AS 136: A K-Means Clustering Algorithm", J. A. Hartigan e M. A. Wong, *Applied Statistics* (nome completo no JSTOR: Journal of the Royal Statistical Society, Series C), vol. 28, nº 1, 1979, primeira página 100.
  - A última página (108) foi conferida no cabeçalho impresso da cópia digitalizada do artigo (https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/HartiganKMeans.pdf, cópia hospedada pela CMU, não pela editora).
  - O artigo só traz as iniciais dos prenomes, então "John A." e "Manchek A." não foram conferidos. Isso não afeta a saída, porque o estilo IEEE abrevia prenomes.
  - `{AS}` protegido ✓.
- **`rousseeuw1987silhouettes`**: no Crossref (https://api.crossref.org/works/10.1016/0377-0427(87)90125-7), *J. Comput. Appl. Math.* 20:53–65, 1987, Peter J. Rousseeuw ✓.
- **`kaufman1990finding`**
  - Na LoC (SRU, ISBN 0471878766, LCCN 89031460): "Finding groups in data: an introduction to cluster analysis", Kaufman e Rousseeuw, New York: Wiley, c1990, série "Wiley series in probability and mathematical statistics" ✓.
  - O Crossref tem a versão online com DOI 10.1002/9780470316801 (https://api.crossref.org/works/10.1002/9780470316801).
- **`hubert1985comparing`**: no Crossref (https://api.crossref.org/works/10.1007/BF01908075), *J. Classification* 2(1):193–218, 1985 ✓.
- **`duda2001pattern`**
  - Na LoC (ISBN 0471056693, LCCN 99029981): Duda, Hart e Stork, *Pattern classification*, 2nd ed., New York: Wiley, c2001 ✓.
  - A LoC registra a editora como "Wiley". A grafia "Wiley-Interscience" do `.bib` não foi conferida em fonte primária; é aceitável, mas "Wiley" acompanharia o catálogo.
- **`hastie2009elements`**
  - Na LoC (ISBN 9780387848570, LCCN 2008941148): 2nd ed., New York, NY: Springer, c2009, série "Springer series in statistics" ✓.
  - No Crossref, o DOI do livro é 10.1007/978-0-387-84858-7 (https://api.crossref.org/works/10.1007/978-0-387-84858-7).
- **`cover1967nearest`**: no Crossref (https://api.crossref.org/works/10.1109/TIT.1967.1053964), *IEEE Trans. Inf. Theory* 13(1):21–27, 1967 ✓.
- **`parzen1962estimation`**: no Crossref (https://api.crossref.org/works/10.1214/aoms/1177704472), *Ann. Math. Statist.* 33(3):1065–1076, 1962 ✓.
- **`silverman1986density`**
  - Na LoC (ISBN 0412246201, LCCN 85021347): London; New York: Chapman and Hall, 1986, série "Monographs on statistics and applied probability" ✓. O número 26 da série aparece no registro da reimpressão de 1998 (LCCN 99019608).
  - O DOI 10.1201/9781315140919 que aparece no Crossref é da **reedição Routledge de 2018** e **não** deve ser usado na entrada de 1986.
- **`ledoit2004wellconditioned`**: no Crossref (https://api.crossref.org/works/10.1016/S0047-259X(03)00096-4), *J. Multivariate Anal.* 88(2):365–411, 2004 ✓.
- **`kuncheva2014combining`**
  - Na LoC (ISBN 9781118315231, LCCN 2014014214): "Second edition", Hoboken, NJ: Wiley, 2014 ✓.
  - No Crossref, o DOI é 10.1002/9781118914564 (https://api.crossref.org/works/10.1002/9781118914564). O campo `edition-number` = 1 do Crossref é inconsistente com o livro, que é a 2ª edição.
- **`varma2006bias`**
  - No Crossref (https://api.crossref.org/works/10.1186/1471-2105-7-91): *BMC Bioinformatics* 7(1), artigo 91, publicado em 2006-02-23 ✓.
  - "91" é **número de artigo**. Com `pages = {91}`, o IEEE imprime "p. 91", o que é aceitável. Opcionalmente, use `eid = {91}`.
- **`cawley2010overfitting`**: no JMLR (https://www.jmlr.org/papers/v11/cawley10a.html), 11(70):2079–2107, 2010 ✓.
- **`dietterich1998approximate`**: no Crossref (https://api.crossref.org/works/10.1162/089976698300017197), *Neural Comput.* 10(7):1895–1923, 1998 ✓.
- **`nadeau2003inference`**: no Crossref (https://api.crossref.org/works/10.1023/A:1024068626366), *Mach. Learn.* 52(3):239–281, 2003 ✓.
- **`bengio2004noUnbiased`**: no JMLR (https://www.jmlr.org/papers/v5/grandvalet04a.html), Bengio e Grandvalet, 5(Sep):1089–1105, 2004 ✓.
- **`friedman1937ranks`**: no Crossref (https://api.crossref.org/works/10.1080/01621459.1937.10503522), *JASA* 32(200):675–701, 1937 ✓.
- **`nemenyi1963distribution`**
  - No catálogo de Princeton (https://catalog.princeton.edu/catalog/9920813653506421, registro MARC em https://bibdata.princeton.edu/bibliographic/9920813653506421): "Nemenyi, Peter Bjorn, 1927-", "Distribution-free multiple comparisons", Ph.D., Princeton University, 1963, orientador John Tukey ✓.
  - Registro ProQuest 302256074 (https://www.proquest.com/docview/302256074), visto apenas por resultado de busca.
  - A referência no Demšar (2006) é idêntica à do `.bib`.
- **`demsar2006statistical`**: no JMLR (https://www.jmlr.org/papers/v7/demsar06a.html), 7(1):1–30, 2006 ✓. `Dem{\v{s}}ar` ✓.
- **`benavoli2016posthoc`**: no JMLR (https://www.jmlr.org/papers/v17/benavoli16a.html), 17(5):1–10, 2016 ✓.
- **`sokolova2009systematic`**: no Crossref (https://api.crossref.org/works/10.1016/j.ipm.2009.03.002), *Inf. Process. Manage.* 45(4):427–437, 2009 ✓.

### Melhorias opcionais (não são erros)

- **`series`/`doi` em livros**, para uniformizar com os artigos:
  - `kaufman1990finding`: `series = {Wiley Series in Probability and Mathematical Statistics}`, `doi = {10.1002/9780470316801}`
  - `hastie2009elements`: `series = {Springer Series in Statistics}`, `doi = {10.1007/978-0-387-84858-7}`
  - `kuncheva2014combining`: `doi = {10.1002/9781118914564}`
  - `silverman1986density`: `series = {Monographs on Statistics and Applied Probability}`, `number = {26}`
- **Números JMLR**, para uniformizar com `benavoli2016posthoc`, que já tem `number = {5}`:
  - `cawley2010overfitting`: `number = {70}`
  - `demsar2006statistical`: `number = {1}`
  - `pedregosa2011scikit`: `number = {85}`
  - `bengio2004noUnbiased`: o JMLR usa "Sep" como fascículo, então é melhor não pôr `number`.
- **`nemenyi1963distribution`**: `address = {Princeton, NJ}` é opcional.

---

## Uso das citações no texto

O contexto de cada `\cite` foi lido (linha anterior, da citação e seguinte). A maioria está bem apoiada. Os pontos abaixo merecem ajuste.

| Linha(s) | Chave(s) | Avaliação |
|---|---|---|
| **353** | `demsar2006statistical` | **Atenção.** A frase atribui ao Demšar o uso de cada fold externo como bloco do teste de Friedman. O Demšar (§2) diz o contrário: os blocos são **bases de dados independentes**, e não amostras dependentes de uma mesma base. A ressalva já aparece na l. 1075, mas aqui a citação sugere que o procedimento vem do artigo. Sugestão: citar `friedman1937ranks` nessa frase ou acrescentar "(ver ressalva na Seção~\ref{sec:friedman})". |
| **688** | `parzen1962estimation` | **Atenção.** Parzen (1962) trata da estimação de densidade **univariada**. O estimador multivariado (kernel produto) é atribuído a Cacoullos (1966): Loftsgaarden e Quesenberry (1965, *Ann. Math. Statist.*, https://projecteuclid.org/euclid.aoms/1177700079, visto por resultado de busca) registram que Cacoullos generalizou o trabalho de Parzen ao caso multivariado. Sugestão: manter Parzen e acrescentar `duda2001pattern` (cap. 4, janelas de Parzen em *d* dimensões; seção não conferida no texto do livro). Outra opção é uma nova entrada Cacoullos, T., "Estimation of a multivariate density", *Ann. Inst. Stat. Math.* 18(1):179–189, 1966, DOI 10.1007/BF02869528 (https://api.crossref.org/works/10.1007/BF02869528). |
| **702** | `hastie2009elements` | **Atenção.** A citação fecha uma frase que inclui o otimizador L-BFGS, o limite de 10 000 iterações e a forma ½‖w‖² + C·Σℓ, que são detalhes da implementação do scikit-learn. O ESL apoia o modelo e a penalização L2, mas não esses detalhes. Sugestão: mover `\cite{hastie2009elements}` para logo após "máxima verossimilhança penalizada" e citar `pedregosa2011scikit` após "10 000 iterações". Isso também aproveita a entrada não usada. |
| **1191** | `kuncheva2014combining` | **Atenção leve.** A inferência de que o desempate a favor da classe 0 explica a maior precisão e a menor cobertura do comitê é dos próprios autores. A obra apoia o voto majoritário e a necessidade de regra de desempate com número par de votantes, mas não essa conclusão específica, e a seção não foi conferida no texto. Sugestão: deixar a citação junto da regra de desempate ou retirá-la dessa frase. |
| 676 | `ledoit2004wellconditioned`, `hastie2009elements` | OK, com nuance. O estimador de Ledoit–Wolf é uma combinação convexa de S com um múltiplo de I, com intensidade estimada dos dados; o ESL (RDA, §4.3.1) faz encolhimento em direção a σ²I. Somar εI com ε fixo é parente próximo dessas regularizações (tipo *ridge*), e o texto já diz "forma mínima". Aceitável. |
| 281–283 | `hopkins1999spambase` | OK. A afiliação Hewlett-Packard vem do arquivo `spambase.DOCUMENTATION` (https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION): criadores no HP Labs em Palo Alto; e-mails não spam vindos de correspondência de trabalho e pessoal arquivada; "george" e "650" como indicadores de não spam. A página atual da UCI só cita a HP por um relatório técnico interno. O texto diz "segundo a documentação da base", o que é correto. Opcional: nomear o arquivo. |
| 225, 386 | `vichi2001double` | OK (é o artigo que introduz o double k-means). |
| 227, 425 | `rousseeuw1987silhouettes` | OK. A definição de s(i), a(i), b(i) e a média sobre os objetos seguem o artigo. |
| 227, 427 | `hubert1985comparing` | OK (é a origem do índice de Rand corrigido). |
| 241 | `hopkins1999spambase` | OK. A página da UCI confirma 4601 instâncias e 57 atributos. |
| 347, 764 | `varma2006bias`, `cawley2010overfitting` | OK. Os dois artigos tratam exatamente do viés otimista de selecionar e avaliar com a mesma validação cruzada. |
| 355 | `kohavi1995study` | OK. Kohavi (pp. 1142–1143) conclui que a estratificação reduz levemente a variância e é, em geral, melhor que a validação cruzada comum em viés e variância. |
| 387 | `govaert2013coclustering` | OK (co-clustering de dados contínuos). |
| 406 | `hartigan1979kmeans` | OK. No AS 136, passo 4a, o ponto só é realocado se o novo custo for **estritamente** menor; em caso de empate, não há realocação (conferido na p. 101 da cópia digitalizada). |
| 491 | `kaufman1990finding` | OK. As faixas (≈0,71–1,00 forte; 0,51–0,70 razoável; 0,26–0,50 fraca; ≤0,25 sem estrutura substancial) batem com o texto, mas foram conferidas **apenas em fontes secundárias** (https://metricgate.com/docs/silhouette-analysis/, https://en.wikipedia.org/wiki/Silhouette_(clustering)). A tabela não foi conferida no livro. Se quiser, acrescente a página no `\cite[p.~…]{}` depois de conferir no livro. |
| 650 | `duda2001pattern` | OK (estimadores de máxima verossimilhança da normal multivariada, com 1/n). |
| 682 | `duda2001pattern`, `cover1967nearest` | OK. A derivação P(ωᵢ\|x) = kᵢ/k é a do Duda (cap. 4), e Cover e Hart é a referência clássica da regra do vizinho mais próximo. |
| 706 | `kuncheva2014combining` | OK (voto majoritário por rótulos). |
| 770 | `sokolova2009systematic` | OK (médias macro). |
| 858 | `dietterich1998approximate`, `nadeau2003inference` | OK. Os dois artigos mostram que tratar resultados de reamostragens sobrepostas como independentes subestima a variância. |
| 948 | `bengio2004noUnbiased` | OK. O teorema principal (página do JMLR) diz que não existe estimador não viesado universal da variância da validação cruzada K-fold. |
| 953 | `friedman1937ranks` | OK. |
| 961 | `nemenyi1963distribution`, `demsar2006statistical` | OK. O Demšar atribui o pós-teste a Nemenyi (1963), e a fórmula da CD é a do artigo. |
| 966 | `demsar2006statistical` | OK. A Tabela 5(a) do Demšar dá q₀,₀₅ = 2,728 para 5 classificadores; CD = 2,728·√(30/1800) ≈ 0,352 ✓. |
| 1075 | `demsar2006statistical` | OK. A ressalva (bases independentes × folds dependentes) está de acordo com o §2 do artigo. |
| 1080 | `benavoli2016posthoc` | OK. A dependência dos testes por postos médios em relação ao conjunto de classificadores é a tese central do artigo. |
| 1138 | `hastie2009elements` | OK (compromisso viés–variância). |
| 1146 | `silverman1986density` | OK. O tamanho amostral exigido cresce muito com a dimensão (discussão conhecida do cap. 4 do livro, que não foi conferida no texto). |
| 1197, 1247 | `nadeau2003inference` | OK (teste *t* corrigido). Sugestão: acrescentar `bouckaert2004evaluating` (ver acima). |

**Ferramentas sem citação:**
- O scikit-learn aparece nas l. 408, 646, 673, 702 e 710 sem citação. A entrada `pedregosa2011scikit` já existe; sugere-se citá-la na l. 646.
- `scipy.stats.friedmanchisquare` (l. 960) e `scikit-posthocs` (l. 967) também não têm citação. Não há entradas para eles no `.bib`; não foram verificados.

---

## Fontes

Crossref (API REST, metadados depositados pelas editoras):
- https://api.crossref.org/works/10.1007/978-3-642-59471-7_6
- https://api.crossref.org/works/10.1007/978-3-642-59471-7
- https://api.crossref.org/works/10.2307/2346830
- https://api.crossref.org/works/10.1016/0377-0427(87)90125-7
- https://api.crossref.org/works/10.1007/BF01908075
- https://api.crossref.org/works/10.1109/TIT.1967.1053964
- https://api.crossref.org/works/10.1214/aoms/1177704472
- https://api.crossref.org/works/10.1016/S0047-259X(03)00096-4
- https://api.crossref.org/works/10.1186/1471-2105-7-91
- https://api.crossref.org/works/10.1162/089976698300017197
- https://api.crossref.org/works/10.1023/A:1024068626366
- https://api.crossref.org/works/10.1007/978-3-540-24775-3_3
- https://api.crossref.org/works/10.1007/b97861
- https://api.crossref.org/works/10.1080/01621459.1937.10503522
- https://api.crossref.org/works/10.1016/j.ipm.2009.03.002
- https://api.crossref.org/works/10.1002/9781118649480
- https://api.crossref.org/works/10.1002/9780470316801
- https://api.crossref.org/works/10.1007/978-0-387-84858-7
- https://api.crossref.org/works/10.1002/9781118914564
- https://api.crossref.org/works/10.1201/9781315140919
- https://api.crossref.org/works/10.1007/BF02869528
- https://api.crossref.org/works/10.24432/C53G6X (respondeu "Resource not found": o DOI é DataCite)
- https://api.crossref.org/works?query.bibliographic=… (buscas pelos livros)

DataCite e UCI:
- https://api.datacite.org/dois/10.24432/C53G6X
- https://archive.ics.uci.edu/dataset/94/spambase
- https://archive.ics.uci.edu/ml/datasets/spambase (redireciona para a página acima)
- https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION
- `dataset/spambase.names` do próprio repositório (arquivo local)

Library of Congress (SRU, registros MODS):
- http://lx2.loc.gov:210/lcdb?version=1.1&operation=searchRetrieve&query=bath.isbn=0471878766&recordSchema=mods (Kaufman e Rousseeuw)
- http://lx2.loc.gov:210/lcdb?version=1.1&operation=searchRetrieve&query=bath.isbn=0471056693&recordSchema=mods (Duda et al.)
- http://lx2.loc.gov:210/lcdb?version=1.1&operation=searchRetrieve&query=bath.isbn=9780387848570&recordSchema=mods (Hastie et al.)
- http://lx2.loc.gov:210/lcdb?version=1.1&operation=searchRetrieve&query=bath.isbn=0412246201&recordSchema=mods (Silverman)
- http://lx2.loc.gov:210/lcdb?version=1.1&operation=searchRetrieve&query=bath.isbn=9781118315231&recordSchema=mods (Kuncheva)
- mesma consulta com ISBN 9781848214736, 1848214731, 9781118649480 e título "co-clustering" (Govaert e Nadif): sem resultados

JMLR:
- https://www.jmlr.org/papers/v11/cawley10a.html
- https://www.jmlr.org/papers/v5/grandvalet04a.html
- https://www.jmlr.org/papers/v7/demsar06a.html
- http://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf (Tabela 5(a), §2 e lista de referências)
- https://www.jmlr.org/papers/v17/benavoli16a.html
- https://www.jmlr.org/papers/v12/pedregosa11a.html

IJCAI:
- https://www.ijcai.org/Proceedings/95-2/
- https://www.ijcai.org/proceedings/1995/147
- https://ijcai.org/proceedings/1995/0147.pdf

Princeton e ProQuest:
- https://catalog.princeton.edu/catalog/9920813653506421 (HTML com desafio anti-robô; usado o `.json`/`.marcxml` do mesmo registro)
- https://bibdata.princeton.edu/bibliographic/9920813653506421
- https://www.proquest.com/docview/302256074 (visto apenas por resultado de busca)

Outros:
- https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/HartiganKMeans.pdf (cópia digitalizada do artigo, hospedada pela CMU)
- https://researchcommons.waikato.ac.nz/handle/10289/1451 (Bouckaert e Frank, repositório institucional)
- https://link.springer.com/book/10.1007%2Fb97861 (LNCS 3056, visto apenas por resultado de busca)
- https://projecteuclid.org/euclid.aoms/1177700079 (Loftsgaarden e Quesenberry 1965, visto apenas por resultado de busca)
- https://www.lehmanns.ch/shop/mathematik-informatik/25704173-9781848214736-co-clustering (listagem comercial, só por busca; 403 no acesso direto)
- https://find.library.upatras.gr/Record/ocn872372294 (catálogo, só por busca; conexão recusada no acesso direto)
- https://metricgate.com/docs/silhouette-analysis/ e https://en.wikipedia.org/wiki/Silhouette_(clustering) (fontes secundárias para a escala de Kaufman e Rousseeuw)

Tentativas bloqueadas (não contornadas):
- https://link.springer.com/chapter/10.1007/978-3-642-59471-7_6 (desafio JavaScript)
- https://onlinelibrary.wiley.com/doi/book/10.1002/9781118649480 (403)
- https://www.wiley.com/en-us/Co-Clustering-p-9781848214736 (403)
- https://www.jstor.org/stable/2346830 (página não carregou)
- https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-33/issue-3/On-Estimation-of-a-Probability-Density-Function-and-Mode/10.1214/aoms/1177704472.full (vazia)
- https://dblp.org/rec/conf/pakdd/BouckaertF04.bib (desafio anti-robô)
- https://www.loc.gov/books/?q=… (desafio Cloudflare; foi usado o SRU)
- https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf (410)
- Google Books API (cota esgotada) e Open Library (sem resultados) para Govaert e Nadif

Arquivos locais lidos (somente leitura): `docs/relatorio/relatorio.tex`, `referencias.bib`, `relatorio.bbl`, `relatorio.blg`, `relatorio.log`, e o estilo `biblatex-ieee` instalado (`/usr/local/texlive/2025/texmf-dist/tex/latex/biblatex-ieee/ieee.bbx`).
