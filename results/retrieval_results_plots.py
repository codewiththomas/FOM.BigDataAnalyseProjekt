#!/usr/bin/env python3
"""
retrieval_results_plots.py

Erzeugt neutrale Matplotlib-Plots aus:
- den GRUPPIERTEN CSVs (aus retrieval_results_grouped.py)
- der aggregierten CSV (retrieval_results.csv)
und zusätzlich KOMBINIERTE Plots, in denen F1 & DSGVO in EINEM Diagramm liegen
(blau/orange) – ideal, um gegenläufige Tendenzen zu erkennen.

Regeln:
- Nur Matplotlib (kein seaborn)
- Ein Chart pro Figure
- Farben nur dort gesetzt, wo explizit gewünscht (kombinierter Plot)

Beispiele:
  python results/retrieval_results_plots.py
  python results/retrieval_results_plots.py --max-bars 30
  python results/retrieval_results_plots.py --group-dir results/tables/retrieval --agg-csv results/tables/retrieval/retrieval_results.csv
  python results/retrieval_results_plots.py --plots-dir results/plots/retrieval
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# Eingaben (Standard-Verzeichnisse)
GROUP_DIR = Path("results/tables/retrieval")
AGG_CSV   = GROUP_DIR / "retrieval_results.csv"

# Zielordner für Plots (gemäß deiner Vorgabe): results/plots/retrieval
PLOTS_DIR = Path("results/plots/retrieval")


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

def combined_combo_plot(agg_csv: Path, plots_dir: Path, max_bars: int = 30):
    """Kombinierter Plot (F1 links, DSGVO rechts) pro Parameter-Kombination."""
    if not agg_csv.exists():
        return
    df = pd.read_csv(agg_csv)

    # Spalten vereinheitlichen (wie zuvor)
    df = df.rename(columns={
        "retrieval_top_k": "top_k",
        "retrieval_similarity_threshold": "similarity_threshold",
        "llm": "llm_model"
    })

    # Label-Bestandteile
    candidates = ["llm_model", "chunking_type", "chunk_size", "top_k", "similarity_threshold"]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return

    def short_chunking(x):
        s = str(x).lower()
        if "fixed" in s: return "fs"
        if "rec" in s:   return "rec"
        if "sem" in s:   return "sem"
        return s

    def short_model(x):
        s = str(x).lower()
        if "mistral" in s:     return "mis7b"
        if "gpt-4o-mini" in s: return "g4o-mini"
        if "qwen" in s:        return "qwen"
        if "llama" in s:       return "llama"
        return str(x)

    def short_thr(v):
        try:
            f = float(v)
            return f"s{int(round(f*100)):02d}"
        except Exception:
            return str(v)

    parts = []
    if "chunking_type" in cols:         parts.append(pd.Series(df["chunking_type"].map(short_chunking), name="ct"))
    if "chunk_size" in cols:            parts.append(pd.Series(df["chunk_size"].astype(str),        name="cs"))
    if "top_k" in cols:                 parts.append(pd.Series("k"+df["top_k"].astype(str),         name="k"))
    if "similarity_threshold" in cols:  parts.append(pd.Series(df["similarity_threshold"].map(short_thr), name="thr"))
    if "llm_model" in cols:             parts.append(pd.Series(df["llm_model"].map(short_model),    name="m"))
    if not parts:
        return

    combo_df = pd.concat(parts, axis=1)
    df["combo"] = combo_df.apply(lambda r: "-".join([str(v) for v in r.values]), axis=1)

    # Mittelwerte je Kombination (volle Genauigkeit, KEIN Runden)
    if "avg_f1" not in df.columns or "avg_dsgvo_score" not in df.columns:
        return
    g = df.groupby("combo", as_index=False)[["avg_f1", "avg_dsgvo_score"]].mean()
    g = g.sort_values("avg_f1", ascending=False).head(max_bars)

    x = np.arange(len(g["combo"]))
    f1_vals    = g["avg_f1"].astype(float).values
    dsgvo_vals = g["avg_dsgvo_score"].astype(float).values

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # F1 links (blau)
    l1 = ax1.plot(x, f1_vals, marker="o", linestyle="-", color="tab:blue", label="F1 (mean)")
    ax1.set_ylabel("F1", color="tab:blue")

    # DSGVO rechts (orange)
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, dsgvo_vals, marker="s", linestyle="-", color="tab:orange", label="DSGVO (mean)")
    ax2.set_ylabel("DSGVO", color="tab:orange")

    # Gemeinsame X-Achse
    labels = g["combo"].astype(str).values
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=6)  # etwas Abstand zwischen Achse und Label
    # etwas unteren Rand reservieren, damit lange vertikale Labels nicht abgeschnitten werden
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)  # ggf. bei sehr vielen Labels auf 0.38–0.45 erhöhen
    ax1.grid(True, axis="y", alpha=0.3)

    # Dynamische Y-Limits:
    f1_max, dsgvo_max = float(np.nanmax(f1_vals)), float(np.nanmax(dsgvo_vals))
    if f1_max <= 0.4 and dsgvo_max <= 0.4:
        ax1.set_ylim(0.0, 0.2)
        ax2.set_ylim(0.0, 0.5)
        major = 0.05
    else:
        # Wenig "Luft" nach oben: +0.05, hart bei 1.0 deckeln
        ax1.set_ylim(0.0, min(1.0, f1_max + 0.05))
        ax2.set_ylim(0.0, min(1.0, dsgvo_max + 0.05))
        # feinere Ticks, je nach Spannweite
        major = 0.05 if max(ax1.get_ylim()[1], ax2.get_ylim()[1]) <= 0.5 else 0.1

    # Mehr Zwischenschritte + 3 Nachkommastellen
    ax1.yaxis.set_major_locator(MultipleLocator(major))
    ax2.yaxis.set_major_locator(MultipleLocator(major))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # Legende kombiniert
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="lower right")

    plt.title(f"F1 & DSGVO (kombiniert) pro Kombination (Top {len(g)})")
    plt.tight_layout()
    out = plots_dir / "combined_f1_dsgvo_by_combination.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)


# === NEU: kombinierter Plot (F1 & DSGVO im selben Chart, blau/orange) ========

def combined_two_metrics(csv_path: Path, index_col: str, outfile: Path, title: str):
    """
    Kombinierter Linienplot: avg_f1__mean (blau) & avg_dsgvo_score__mean (orange)
    im selben Chart – beide 0..1, daher gleiche y-Achse.
    """
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    f1_col   = "avg_f1__mean"
    dsgvo_col= "avg_dsgvo_score__mean"
    if index_col not in df.columns or f1_col not in df.columns or dsgvo_col not in df.columns:
        return

    # sinnvolle Sortierung (numerisch, falls möglich)
    try:
        order = np.argsort(pd.to_numeric(df[index_col], errors="coerce").fillna(np.inf).values)
        df = df.iloc[order]
    except Exception:
        pass

    x = np.arange(len(df[index_col]))
    f1 = df[f1_col].astype(float).values
    ds = df[dsgvo_col].astype(float).values

    plt.figure(figsize=(9.5, 5))
    # Farben explizit wie gewünscht
    plt.plot(x, f1, marker="o", linestyle="-", color="tab:blue",  label="F1 (mean)")
    plt.plot(x, ds, marker="s", linestyle="-", color="tab:orange", label="DSGVO (mean)")
    plt.xticks(x, df[index_col].astype(str).values, rotation=0)
    plt.ylim(0, 1)  # beide Metriken 0..1
    plt.grid(True, axis="y", alpha=0.3)
    plt.title(title)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Erzeuge neutrale Plots (inkl. kombinierter F1+DSGVO) aus gruppierten und aggregierten Retrieval-Tabellen.")
    ap.add_argument("--group-dir", type=Path, default=GROUP_DIR, help="Ordner mit gruppierten CSVs (Default: results/tables/retrieval)")
    ap.add_argument("--agg-csv", type=Path, default=AGG_CSV, help="Aggregierte Ergebnis-CSV (Default: results/tables/retrieval/retrieval_results.csv)")
    ap.add_argument("--max-bars", type=int, default=30, help="Maximale Anzahl Balken bei Kombinationsplots (Default: 30)")
    ap.add_argument("--plots-dir", type=Path, default=PLOTS_DIR, help="Zielordner für Plots (Default: results/plots/retrieval)")
    args = ap.parse_args()

    d = args.group_dir
    plots = args.plots_dir

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
    combined_combo_plot(args.agg_csv, plots, max_bars=args.max_bars)

    # 4) NEU: Kombinierte Linienplots (F1 & DSGVO im selben Chart)
    combined_two_metrics(d / "retrieval_grouped_by_chunk_size.csv", "chunk_size",
                         plots / "combined_f1_dsgvo_by_chunk_size.png",
                         "F1 & DSGVO (kombiniert) nach Chunk Size")
    combined_two_metrics(d / "retrieval_grouped_by_chunking_type.csv", "chunking_type",
                         plots / "combined_f1_dsgvo_by_chunking_type.png",
                         "F1 & DSGVO (kombiniert) nach Chunking-Methode")
    combined_two_metrics(d / "retrieval_grouped_by_top_k.csv", "top_k",
                         plots / "combined_f1_dsgvo_by_top_k.png",
                         "F1 & DSGVO (kombiniert) nach top_k")
    combined_two_metrics(d / "retrieval_grouped_by_similarity_threshold.csv", "similarity_threshold",
                         plots / "combined_f1_dsgvo_by_threshold.png",
                         "F1 & DSGVO (kombiniert) nach Threshold")
    combined_two_metrics(d / "retrieval_grouped_by_llm_model.csv", "llm_model",
                         plots / "combined_f1_dsgvo_by_llm.png",
                         "F1 & DSGVO (kombiniert) nach LLM-Modell")

    print(f"✅ Plots gespeichert unter: {plots}")


if __name__ == "__main__":
    main()
