#!/usr/bin/env python3
"""
retrieval_results_plots.py

Liest gruppierte CSVs (aus retrieval_results_grouped.py) sowie die aggregierte
Tabelle (retrieval_results.csv) und erzeugt neutrale Matplotlib-Plots.

Regeln:
- Nur Matplotlib (kein seaborn)
- Ein Chart pro Figure
- Keine Farben explizit setzen

Beispiele:
  python results/retrieval_results_plots.py
  python results/retrieval_results_plots.py --max-bars 30
  python results/retrieval_results_plots.py --group-dir results/tables/retrieval --agg-csv results/tables/retrieval/retrieval_results.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUP_DIR = Path("results/tables/retrieval")
AGG_CSV   = GROUP_DIR / "retrieval_results.csv"
PLOTS_DIR = GROUP_DIR / "plots"


def bar_with_error(csv_path: Path, index_col: str, metric: str, outfile: Path, title: str):
    """Balkendiagramm mit Fehlerbalken (mean ± std) aus einer gruppierten CSV."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    mean_col = f"{metric}__mean"
    std_col  = f"{metric}__std"
    if index_col not in df.columns or mean_col not in df.columns:
        return

    x = np.arange(len(df[index_col]))
    y = df[mean_col].values
    yerr = df[std_col].values if std_col in df.columns else None

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, y, yerr=yerr)
    plt.xticks(x, df[index_col].astype(str).values, rotation=0)
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def heatmap_two_factor(csv_path: Path, row: str, col: str, metric: str, outfile: Path, title: str):
    """Heatmap aus einer 2-Faktor-Gruppe (z. B. top_k × similarity_threshold)."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    mean_col = f"{metric}__mean"
    if row not in df.columns or col not in df.columns or mean_col not in df.columns:
        return

    pivot = df.pivot_table(index=row, columns=col, values=mean_col)
    arr = pivot.values.astype(float)

    plt.figure(figsize=(6, 4.5))
    plt.imshow(arr, aspect='auto')
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=0)
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def build_combo_plots(agg_csv: Path, plots_dir: Path, max_bars: int = 30):
    """
    Balkendiagramme „pro Kombination“ aus der aggregierten Ergebnis-CSV.
    Kombination = (llm_model, chunking_type, chunk_size, top_k, similarity_threshold), soweit vorhanden.
    """
    if not agg_csv.exists():
        return
    df = pd.read_csv(agg_csv)

    # Spalten vereinheitlichen
    df = df.rename(columns={
        "retrieval_top_k": "top_k",
        "retrieval_similarity_threshold": "similarity_threshold",
        "llm": "llm_model"
    })

    # Relevante Spalten für die Kombinationskennung
    candidates = ["llm_model", "chunking_type", "chunk_size", "top_k", "similarity_threshold"]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return

    # Kurze Label-Heuristiken
    def short_chunking(x):
        s = str(x).lower()
        if "fixed" in s: return "fs"
        if "rec" in s: return "rec"
        if "sem" in s: return "sem"
        return s

    def short_model(x):
        s = str(x).lower()
        if "mistral" in s: return "mis7b"
        if "gpt-4o-mini" in s: return "g4o-mini"
        if "qwen" in s: return "qwen"
        if "llama" in s: return "llama"
        return str(x)

    def short_thr(v):
        try:
            f = float(v)
            return f"s{int(round(f*100)):02d}"
        except Exception:
            return str(v)

    parts = []
    if "chunking_type" in cols: parts.append(pd.Series(df["chunking_type"].map(short_chunking), name="ct"))
    if "chunk_size" in cols:    parts.append(pd.Series(df["chunk_size"].astype(str), name="cs"))
    if "top_k" in cols:         parts.append(pd.Series("k"+df["top_k"].astype(str), name="k"))
    if "similarity_threshold" in cols: parts.append(pd.Series(df["similarity_threshold"].map(short_thr), name="thr"))
    if "llm_model" in cols:     parts.append(pd.Series(df["llm_model"].map(short_model), name="m"))

    if not parts:
        return

    combo_df = pd.concat(parts, axis=1)
    df["combo"] = combo_df.apply(lambda r: "-".join([str(v) for v in r.values]), axis=1)

    metrics = [m for m in ["avg_f1", "avg_dsgvo_score"] if m in df.columns]
    if not metrics:
        return

    grouped = df.groupby("combo", as_index=False)[metrics].mean()

    for metric in metrics:
        g_sorted = grouped.sort_values(metric, ascending=False).head(max_bars)
        x = np.arange(len(g_sorted["combo"]))
        y = g_sorted[metric].values

        plt.figure(figsize=(12, 5))
        plt.bar(x, y)
        plt.xticks(x, g_sorted["combo"].astype(str).values, rotation=90)
        plt.ylabel(metric)
        plt.title(f"{metric} pro Kombination (Top {min(max_bars, len(g_sorted))})")
        plt.tight_layout()
        out = plots_dir / f"{metric}_by_combination.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Erzeuge neutrale Plots aus gruppierten und aggregierten Retrieval-Tabellen.")
    ap.add_argument("--group-dir", type=Path, default=GROUP_DIR, help="Ordner mit gruppierten CSVs (Default: results/tables/retrieval)")
    ap.add_argument("--agg-csv", type=Path, default=AGG_CSV, help="Aggregierte Ergebnis-CSV (Default: results/tables/retrieval/retrieval_results.csv)")
    ap.add_argument("--max-bars", type=int, default=30, help="Maximale Anzahl Balken bei Kombinationsplots (Default: 30)")
    args = ap.parse_args()

    d = args.group_dir
    plots = d / "plots"

    # 1) Einfache Balkenplots (mean ± std) aus gruppierten Tabellen
    bar_with_error(d / "retrieval_grouped_by_chunk_size.csv", "chunk_size", "avg_f1", plots / "f1_by_chunk_size.png", "F1 nach Chunk Size")
    bar_with_error(d / "retrieval_grouped_by_chunk_size.csv", "chunk_size", "avg_dsgvo_score", plots / "dsgvo_by_chunk_size.png", "DSGVO-Score nach Chunk Size")

    bar_with_error(d / "retrieval_grouped_by_chunking_type.csv", "chunking_type", "avg_f1", plots / "f1_by_chunking_type.png", "F1 nach Chunking-Methode")
    bar_with_error(d / "retrieval_grouped_by_chunking_type.csv", "chunking_type", "avg_dsgvo_score", plots / "dsgvo_by_chunking_type.png", "DSGVO-Score nach Chunking-Methode")

    bar_with_error(d / "retrieval_grouped_by_top_k.csv", "top_k", "avg_f1", plots / "f1_by_top_k.png", "F1 nach top_k")
    bar_with_error(d / "retrieval_grouped_by_top_k.csv", "top_k", "avg_dsgvo_score", plots / "dsgvo_by_top_k.png", "DSGVO-Score nach top_k")

    bar_with_error(d / "retrieval_grouped_by_similarity_threshold.csv", "similarity_threshold", "avg_f1", plots / "f1_by_threshold.png", "F1 nach Threshold")
    bar_with_error(d / "retrieval_grouped_by_similarity_threshold.csv", "similarity_threshold", "avg_dsgvo_score", plots / "dsgvo_by_threshold.png", "DSGVO-Score nach Threshold")

    bar_with_error(d / "retrieval_grouped_by_llm_model.csv", "llm_model", "avg_f1", plots / "f1_by_llm.png", "F1 nach LLM-Modell")
    bar_with_error(d / "retrieval_grouped_by_llm_model.csv", "llm_model", "avg_dsgvo_score", plots / "dsgvo_by_llm.png", "DSGVO-Score nach LLM-Modell")

    # 2) Heatmaps: top_k × threshold
    heatmap_two_factor(d / "retrieval_grouped_by_topk_and_threshold.csv", "top_k", "similarity_threshold", "avg_f1", plots / "heatmap_f1_topk_threshold.png", "F1: top_k × threshold")
    heatmap_two_factor(d / "retrieval_grouped_by_topk_and_threshold.csv", "top_k", "similarity_threshold", "avg_dsgvo_score", plots / "heatmap_dsgvo_topk_threshold.png", "DSGVO: top_k × threshold")

    # 3) „Pro Kombination“ (aus der aggregierten CSV)
    build_combo_plots(args.agg_csv, plots, max_bars=args.max_bars)

    print(f"✅ Plots gespeichert unter: {plots}")


if __name__ == "__main__":
    main()
