
#!/usr/bin/env python3
"""
retrieval_results_grouped.py
Erzeugt GRUPPIERTE, tabellarische Ausgaben (CSV + optional Markdown + optional Console-Print)
aus der aggregierten Retrieval-CSV oder direkt aus JSON-Summaries.

- Standardquelle: results/tables/retrieval/retrieval_results.csv
- Fallback:      results/runs/**/*_evaluation_summary_*.json

Nutzt ausschließlich pandas/matplotlib-neutrale Tabellen (keine Farben).

CLI-Beispiele:
  # Standard (CSV + optional MD, zusätzlich Konsolen-Print)
  python results/retrieval_results_grouped.py --print

  # Ohne Markdown
  python results/retrieval_results_grouped.py --no-md --print

  # Alle Gruppentabellen zusätzlich in eine Excel-Datei bündeln
  python results/retrieval_results_grouped.py --print --excel results/tables/retrieval/retrieval_groups.xlsx

"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import argparse, json, sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR  = PROJECT_ROOT / "results" / "runs" / "retrieval"
OUT_DIR   = PROJECT_ROOT / "results" / "tables" / "retrieval"
CSV_PATH  = OUT_DIR / "retrieval_results.csv"

def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def norm_chunking_type(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    low = str(name).lower()
    if "semantic" in low:
        return "semantic"
    if "recursive" in low:
        return "recursive"
    if "fixed" in low:
        return "fixedsize"
    return name

def collect_results(summary_dir: Path) -> pd.DataFrame:
    records = []
    files = sorted(summary_dir.glob("**/*_evaluation_summary_*.json"))
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        records.append({
            "experiment_file": p.name,
            "llm_model": safe_get(data, "pipeline_info.llm.model") or safe_get(data, "pipeline_info.llm.name"),
            "embedding_model": safe_get(data, "pipeline_info.embedding.model") or safe_get(data, "pipeline_info.embedding.name"),
            "chunking_type": norm_chunking_type(safe_get(data, "pipeline_info.chunking.name")),
            "chunk_size": safe_get(data, "pipeline_info.chunking.chunk_size") or safe_get(data, "pipeline_info.chunking.strategy"),
            "chunk_overlap": safe_get(data, "pipeline_info.chunking.chunk_overlap"),
            "top_k": safe_get(data, "pipeline_info.retrieval.top_k"),
            "similarity_threshold": safe_get(data, "pipeline_info.retrieval.similarity_threshold"),
            "avg_precision": data.get("avg_precision"),
            "avg_recall": data.get("avg_recall"),
            "avg_f1": data.get("avg_f1"),
            "avg_dsgvo_score": data.get("avg_dsgvo_score"),
            "qa_pairs_evaluated": data.get("qa_pairs_evaluated"),
        })
    return pd.DataFrame.from_records(records)

def load_base_table(in_csv: Optional[Path], runs_dir: Path) -> pd.DataFrame:
    if in_csv and in_csv.exists():
        return pd.read_csv(in_csv)
    return collect_results(runs_dir)

def group_and_format(df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
    metrics = [m for m in ["avg_f1","avg_dsgvo_score","avg_precision","avg_recall"] if m in df.columns]
    if not metrics:
        return pd.DataFrame()
    g = (df.groupby(by_cols, dropna=False)[metrics]
           .agg(['mean','std','count'])
           .reset_index())
    # Spaltennamen auf eine Ebene bringen
    g.columns = ['__'.join(filter(None, map(str, col))).strip('_') for col in g.columns.values]
    # Runden & Typen setzen
    for m in metrics:
        mean_col = f"{m}__mean"
        std_col  = f"{m}__std"
        cnt_col  = f"{m}__count"
        if mean_col in g.columns:
            g[mean_col] = g[mean_col].astype(float).round(3)
        if std_col in g.columns:
            g[std_col] = g[std_col].fillna(0.0).astype(float).round(3)
        if cnt_col in g.columns:
            g[cnt_col] = g[cnt_col].astype(int)
    return g

def save_table(df: pd.DataFrame, out_path: Path, also_markdown: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    if also_markdown:
        try:
            md_path = out_path.with_suffix(".md")
            df.to_markdown(md_path, index=False)
        except Exception:
            pass

def print_table(df: pd.DataFrame, title: str) -> None:
    # Kompakte, gut lesbare Ausgabe für PyCharm-Konsole
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    with pd.option_context('display.max_rows', 200, 'display.max_columns', 50, 'display.width', 180):
        print(df)

def main():
    ap = argparse.ArgumentParser(description="Gruppierte Auswertung tabellarisch erzeugen (CSV/MD + Console Print).")
    ap.add_argument("--csv", type=Path, default=CSV_PATH, help="Pfad zur aggregierten CSV (Default: results/tables/retrieval/retrieval_results.csv)")
    ap.add_argument("--runs", type=Path, default=RUNS_DIR, help="Fallback: Verzeichnis mit JSON-Summaries (Default: results/runs)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="Zielordner (Default: results/tables/retrieval)")
    ap.add_argument("--no-md", action="store_true", help="Kein Markdown speichern (nur CSV)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Tabellen zusätzlich in die Konsole drucken (PyCharm Ansicht)")
    ap.add_argument("--excel", type=Path, default=None, help="Optional: Pfad zu einer Excel-Datei, in die alle Gruppentabellen als einzelne Sheets geschrieben werden (z. B. results/tables/retrieval/retrieval_groups.xlsx)")
    args = ap.parse_args()

    df = load_base_table(args.csv, args.runs)
    if df.empty:
        raise SystemExit("Keine Daten gefunden.")

    # Harmonisiere Spaltennamen, falls aus CSV
    df = df.rename(columns={
        "retrieval_top_k":"top_k",
        "retrieval_similarity_threshold":"similarity_threshold",
        "llm":"llm_model"
    })

    groupings = [
        (["llm_model"],                           "retrieval_grouped_by_llm_model"),
        (["chunking_type"],                       "retrieval_grouped_by_chunking_type"),
        (["chunk_size"],                          "retrieval_grouped_by_chunk_size"),
        (["top_k"],                               "retrieval_grouped_by_top_k"),
        (["similarity_threshold"],                "retrieval_grouped_by_similarity_threshold"),
        (["chunking_type","chunk_size"],          "retrieval_grouped_by_chunking_type_and_size"),
        (["chunking_type","top_k"],               "retrieval_grouped_by_chunking_type_and_top_k"),
        (["top_k","similarity_threshold"],        "retrieval_grouped_by_topk_and_threshold"),
        (["llm_model","chunking_type"],           "retrieval_grouped_by_llm_and_chunking_type"),
        (["llm_model","chunk_size"],              "retrieval_grouped_by_llm_and_chunk_size"),
        (["llm_model","top_k","similarity_threshold"], "retrieval_grouped_by_llm_topk_threshold"),
    ]

    produced = []  # (prefix, dataframe)
    for by_cols, prefix in groupings:
        existing = [c for c in by_cols if c in df.columns]
        if not existing:
            continue
        table = group_and_format(df, existing)
        if table.empty:
            continue
        out_path = args.out / f"{prefix}.csv"
        save_table(table, out_path, also_markdown=not args.no_md)
        produced.append((prefix, table))
        if args.do_print:
            print_table(table, f"{prefix}")

    # Optional: alles in eine Excel-Datei schreiben (mehrere Sheets)
    if args.excel:
        args.excel.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(args.excel, engine='xlsxwriter') as writer:
            for prefix, table in produced:
                # Sheetnamen auf 31 Zeichen begrenzen und unzulässige Zeichen entfernen
                sheet = prefix[:31].replace(':','_').replace('/','_').replace('\\','_').replace('*','_').replace('?','_').replace('[','(').replace(']',')')
                table.to_excel(writer, sheet_name=sheet, index=False)
        print(f"📘 Excel exportiert: {args.excel}")

    print(f"✅ Gruppierte Tabellen gespeichert unter: {args.out}")

if __name__ == "__main__":
    main()
