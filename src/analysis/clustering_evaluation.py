"""Complete the Q1 analysis without refitting Double k-means."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

from src.analysis.silhouette import _load_analysis_inputs, analyze_silhouette
from src.experiments.question_1 import _atomic_output
from src.utils.dataloader import load_spambase_original
from src.visualization.q1_plots import plot_q1_results


LOGGER = logging.getLogger(__name__)


def _markdown_table(frame):
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.8f}" if isinstance(value, float) else str(value)
                  for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _ari_comment(ari):
    if np.isclose(ari, 1, rtol=0, atol=1e-12):
        return "As particoes coincidem, desconsiderando a numeracao dos clusters."
    if ari < 0:
        return (
            "A concordancia ajustada e negativa: esta abaixo da esperada ao acaso "
            "sob o modelo de referencia do ARI. Os clusters nao recuperam a "
            "particao spam/nao spam."
        )
    if np.isclose(ari, 0, rtol=0, atol=1e-12):
        return "A concordancia ajustada esta no nivel esperado ao acaso pelo ARI."
    return (
        "Ha concordancia positiva acima da referencia ao acaso, mas as particoes "
        "nao coincidem. O valor deve ser interpretado junto da tabela de contingencia."
    )


def _validate_winner(best, X):
    K, H = best["K"], best["H"]
    columns = np.asarray(best["column_labels"])
    G = np.asarray(best["prototypes"])
    values = np.asarray(best["objective_history"])
    if (columns.shape != (X.shape[1],)
            or not np.issubdtype(columns.dtype, np.integer)
            or not np.array_equal(np.unique(columns), np.arange(H))
            or G.shape != (K, H) or not np.isfinite(G).all()
            or values.shape != (best["n_iter"] + 1,)
            or not np.isfinite(values).all()
            or not np.isclose(values[-1], best["objective"], rtol=1e-12)):
        raise ValueError("Invalid winning prototypes, column labels or objective history.")
    if np.any(np.diff(values) > 1e-12 * max(1.0, abs(values[0]))):
        raise ValueError("Winning objective history is not non-increasing.")
    reconstructed = G[np.ix_(best["labels"], columns)]
    if not np.isclose(np.sum((X - reconstructed) ** 2), best["objective"], rtol=1e-12):
        raise ValueError("Winning objective does not match the saved partition.")


def _write_report(path, metrics, scores, contingency, sizes, column_sizes, G,
                  plots_dir, results_dir):
    def link(target):
        return Path(os.path.relpath(target, path.parent)).as_posix()

    ari = metrics["ari"]
    text = f"""# Analise da Questao 1

## Protocolo e dados

- Spambase: {metrics['n_samples']} observacoes, {metrics['n_features']} atributos;
  a ultima coluna contem a classe original (0 = nao spam, 1 = spam).
- Padronizacao z-score por atributo no dataset completo, sem reordenar linhas.
  A classe original nao participa do ajuste nem da selecao por silhueta.
- {metrics['n_runs_per_pair']} inicializacoes por par, em nove pares (K, H):
  {metrics['total_runs']} ajustes. Sementes de 0 a {metrics['n_runs_per_pair'] - 1}.
- Em cada par, menor objetivo W; empate exato resolvido pela primeira semente.
  Entre pares, maior silhueta euclidiana completa; empate por menor K e depois H.
- Hash SHA-256 do CSV: `{metrics['dataset_sha256']}`.
- Convergiram {metrics['converged_runs']} de {metrics['total_runs']} ajustes.
  Limite de iteracoes: {metrics['max_iter']}.

{"**ATENCAO: experimento reduzido; nao substitui os 100 ajustes por par exigidos.**" if metrics['n_runs_per_pair'] != 100 else ""}

## Comparacao por silhueta

{_markdown_table(scores)}

Selecionado: **K*={metrics['K']}, H*={metrics['H']}, semente={metrics['best_run_id']}**,
com silhueta **{metrics['silhouette']:.8f}** e W={metrics['best_objective']:.8f}.
O menor objetivo global nao e o criterio para escolher K*: W e usado primeiro
dentro de cada par. A silhueta avalia separacao geometrica na escala padronizada,
nao a recuperacao das classes originais.

![Silhueta]({link(plots_dir / 'silhouette_heatmap.png')})

## Concordancia com a particao a priori

**ARI = {ari:.10f}.** {_ari_comment(ari)}
ARI=1 indica particoes identicas a menos de permutacao; ARI proximo de zero
indica concordancia proxima da referencia ao acaso. Valores negativos sao possiveis.

Tabela de contingencia (linhas = classes originais, colunas = clusters):

{_markdown_table(contingency.reset_index())}

Os IDs dos clusters sao arbitrarios. Nao se interpreta a diagonal como acertos
de classificacao nem se renomeiam clusters como spam/nao spam.
A tabela e retangular quando ha mais clusters do que classes.

![Contingencia]({link(plots_dir / 'confusion_matrix_q1.png')})

## Distribuicao dos clusters

{_markdown_table(sizes)}

Maior cluster: {metrics['largest_cluster_fraction']:.2%} das observacoes.
Uma silhueta elevada pode coexistir com forte desequilibrio dos grupos e ARI
baixo; por isso a silhueta nao demonstra, sozinha, separacao de spam e nao spam.
As composicoes por classe na tabela acima devem ser examinadas junto dos tamanhos.
Nao se altera o K* selecionado a posteriori com base nas classes originais.

Grupos de variaveis:

{_markdown_table(column_sizes)}

## Prototipos e convergencia

G contem as medias dos blocos na escala z-score, nao nas unidades originais:

{_markdown_table(G.reset_index())}

![Prototipos]({link(plots_dir / 'prototypes_heatmap.png')})

Execucao vencedora: {metrics['n_iterations']} iteracoes;
convergencia: {metrics['best_converged']}.
W inicial={metrics['initial_objective']:.8f};
W final={metrics['best_objective']:.8f}.
A curva inclui a inicializacao (iteracao 0) e o valor ao fim de cada iteracao.

![Convergencia]({link(plots_dir / 'objective_convergence.png')})

## Reprodutibilidade e proxima etapa

As analises reutilizam o historico, sem novos ajustes do clustering.
Os resultados completos e a interpretacao devem ser preservados junto do codigo:

- [Metricas]({link(results_dir / 'q1_evaluation.json')})
- [Scores]({link(results_dir / 'silhouette_scores.csv')})
- [Contingencia]({link(results_dir / 'q1_contingency.csv')})
- [Particao selecionada]({link(results_dir / 'q1_selected_partition.npz')})
- [Rotulos por linha]({link(results_dir / 'q1_object_labels.csv')})

Para a Q2, usar exatamente os rotulos da particao selecionada na ordem original
das linhas. Mesmo quando K*=2, esses rotulos nao sao as classes originais.
Nao confundir desempenho futuro de classificacao desses clusters com deteccao
de spam. A integracao do dataloader e a validacao cruzada da Q2 nao fazem parte
desta etapa.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_output(path) as stream:
        stream.write(text.encode("utf-8"))


def evaluate_clustering(filepath="dataset/spambase.data", results_dir="results",
                        plots_dir="plots", report_path="analysis/Q1_ANALYSIS.md"):
    """Regenerate silhouettes, evaluate their winner, and save all Q1 analysis."""
    results_dir, plots_dir = Path(results_dir), Path(plots_dir)
    scores = analyze_silhouette(filepath, results_dir, plots_dir)
    X, history = _load_analysis_inputs(Path(filepath), results_dir / "q1_full_history.pkl")
    selection = scores.loc[scores["silhouette"].idxmax()]
    runs = [run for run in history["runs"]
            if (run["K"], run["H"]) == (int(selection.K), int(selection.H))]
    best = min(runs, key=lambda run: run["objective"])
    _validate_winner(best, X)
    _, y = load_spambase_original(filepath)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Spambase original classes must contain both 0 and 1.")

    labels = best["labels"]
    classes = np.unique(y).astype(int)
    contingency = pd.DataFrame(
        contingency_matrix(y, labels), index=classes, columns=np.arange(best["K"])
    ).rename_axis(index="original_class", columns="cluster")
    counts = np.bincount(labels, minlength=best["K"])
    sizes = pd.DataFrame({
        "cluster": np.arange(best["K"]), "n_objects": counts,
        "fraction": counts / len(y),
    })
    column_sizes = pd.DataFrame({
        "variable_cluster": np.arange(best["H"]),
        "n_features": np.bincount(best["column_labels"], minlength=best["H"]),
    })
    G = pd.DataFrame(
        best["prototypes"], columns=[f"variable_cluster_{h}" for h in range(best["H"])]
    ).rename_axis("object_cluster")
    objective = pd.DataFrame({
        "iteration": np.arange(len(best["objective_history"])),
        "objective": best["objective_history"],
    })
    metrics = {
        "K": best["K"], "H": best["H"], "best_run_id": best["run"],
        "silhouette": float(selection.silhouette),
        "best_objective": best["objective"], "best_converged": bool(best["converged"]),
        "n_iterations": best["n_iter"],
        "initial_objective": float(best["objective_history"][0]),
        "ari": float(adjusted_rand_score(y, labels)),
        "n_samples": X.shape[0], "n_features": X.shape[1],
        "n_runs_per_pair": history["metadata"]["n_runs"],
        "max_iter": history["metadata"]["max_iter"],
        "total_runs": len(history["runs"]),
        "converged_runs": sum(bool(run["converged"]) for run in history["runs"]),
        "largest_cluster_fraction": float(counts.max() / len(y)),
        "dataset_sha256": history["metadata"]["dataset_sha256"],
    }

    plot_q1_results(best, contingency, plots_dir)
    frames = {
        "q1_contingency.csv": contingency.reset_index(),
        "q1_cluster_sizes.csv": sizes,
        "q1_variable_cluster_sizes.csv": column_sizes,
        "q1_prototypes.csv": G.reset_index(),
        "q1_objective_history.csv": objective,
        "q1_object_labels.csv": pd.DataFrame({
            "row_index": np.arange(len(y)), "original_class": y.astype(int), "cluster": labels,
        }),
        "q1_variable_labels.csv": pd.DataFrame({
            "feature_index": np.arange(X.shape[1]), "cluster": best["column_labels"],
        }),
    }
    for name, frame in frames.items():
        with _atomic_output(results_dir / name) as stream:
            stream.write(frame.to_csv(index=False).encode("utf-8"))
    with _atomic_output(results_dir / "q1_selected_partition.npz") as stream:
        np.savez_compressed(
            stream, **best, scaler_mean=history["scaler_mean"],
            scaler_scale=history["scaler_scale"],
            dataset_sha256=history["metadata"]["dataset_sha256"],
        )
    with _atomic_output(results_dir / "q1_evaluation.json") as stream:
        stream.write(json.dumps(metrics, indent=2, allow_nan=False).encode("utf-8"))
    with _atomic_output(results_dir / "ari_score.txt") as stream:
        stream.write(f"ARI = {metrics['ari']:.10f}\n{_ari_comment(metrics['ari'])}\n".encode())
    _write_report(Path(report_path), metrics, scores, contingency, sizes,
                  column_sizes, G, plots_dir, results_dir)
    LOGGER.info("Q1 evaluated: K=%d H=%d ARI=%.10f", best["K"], best["H"], metrics["ari"])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/spambase.data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--report-path", default="analysis/Q1_ANALYSIS.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = evaluate_clustering(args.data, args.results_dir, args.plots_dir, args.report_path)
    print(f"K*={metrics['K']}, H*={metrics['H']}, ARI={metrics['ari']:.10f}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
