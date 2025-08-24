#!/usr/bin/env python3

# Standard (CSV + optional MD, zusätzlich Konsolen-Print)
# python results/retrieval_results_table.py

"""
Aggregiert alle *_evaluation_summary_*.json zu einer Tabelle.
Liest ausschließlich aus dem Dateiinhalt (pipeline_info + avg_* Metrics).
Speichert CSV und Markdown in results/tables/.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json

import pandas as pd


# Dieses Skript liegt unter: results/retrieval_results_table.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR  = PROJECT_ROOT / "results" / "runs" / "retrieval"
OUT_DIR      = PROJECT_ROOT / "results" / "tables" / "retrieval"


def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def norm_chunking_type(name: Optional[str]) -> Optional[str]:
    """Normalisiert die Chunking-Bezeichnung aus pipeline_info.chunking.name."""
    if not name:
        return None
    low = name.lower()
    if "semantic" in low:
        return "semantic"
    if "recursive" in low:
        return "recursive"
    if "fixed" in low:
        return "fixedsize"
    return name  # fallback: Rohwert


def model_abbr_from_embedding(model: Optional[str]) -> Optional[str]:
    """Heuristische Abkürzung für Embedding-Modell (falls in JSON keine Abkürzung steht)."""
    if not model:
        return None
    m = model.lower()
    if "all-minilm-l6-v2" in m:
        return "st-mini"
    if "all-mpnet-base-v2" in m:
        return "mpnet"
    return None


def collect_results(summary_dir: Path) -> pd.DataFrame:
    records = []
    files = sorted(summary_dir.glob("*_evaluation_summary_*.json"))
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Pipeline-Infos (robust lesen) ---
        llm_name   = safe_get(data, "pipeline_info.llm.model") or safe_get(data, "pipeline_info.llm.name")
        llm_temp   = safe_get(data, "pipeline_info.llm.temperature")
        llm_tokens = safe_get(data, "pipeline_info.llm.max_tokens")

        emb_model  = safe_get(data, "pipeline_info.embedding.model") or safe_get(data, "pipeline_info.embedding.name")
        emb_abbr   = safe_get(data, "pipeline_info.embedding.abbr") or model_abbr_from_embedding(emb_model)

        ch_name    = safe_get(data, "pipeline_info.chunking.name")
        ch_type    = norm_chunking_type(ch_name)
        ch_size    = safe_get(data, "pipeline_info.chunking.chunk_size") or safe_get(data, "pipeline_info.chunking.strategy")
        ch_overlap = safe_get(data, "pipeline_info.chunking.chunk_overlap")

        top_k      = safe_get(data, "pipeline_info.retrieval.top_k")
        sim_thr    = safe_get(data, "pipeline_info.retrieval.similarity_threshold")

        max_ctx    = safe_get(data, "pipeline_info.pipeline.max_context_length")
        grouping   = safe_get(data, "pipeline_info.dataset.grouping_applied")

        # --- Metriken (IR + RAGAS/DSGVO + Performance) ---
        rec = {
            "experiment_file": p.name,
            # LLM
            "llm_model": llm_name,
            "llm_temperature": llm_temp,
            "llm_max_tokens": llm_tokens,
            # Embedding
            "embedding_model": emb_model,
            # Chunking/Retrieval
            "chunking_type": ch_type,
            "chunk_size": ch_size,
            "chunk_overlap": ch_overlap,
            "top_k": top_k,
            "similarity_threshold": sim_thr,
            "grouping": grouping,
            "max_context_length": max_ctx,
            # IR
            "avg_precision": data.get("avg_precision"),
            "avg_recall": data.get("avg_recall"),
            "avg_f1": data.get("avg_f1"),
            # RAGAS + DSGVO
            "avg_faithfulness": data.get("avg_faithfulness"),
            "avg_answer_relevance": data.get("avg_answer_relevance"),
            "avg_context_relevance": data.get("avg_context_relevance"),
            "avg_dsgvo_score": data.get("avg_dsgvo_score"),
            # Performance
            "avg_query_time": data.get("avg_query_time"),
            "avg_response_length": data.get("avg_response_length"),
            # Validation / Kontext-Diagnostik
            "chunks_indexed": safe_get(data, "pipeline_info.retrieval.chunks_indexed"),
            "truncation_rate": safe_get(data, "context_optimization.truncation_rate"),
            "avg_context_utilization": safe_get(data, "context_optimization.avg_context_utilization"),
            "avg_chunks_used": safe_get(data, "context_optimization.avg_chunks_used"),
            "total_chunks_wasted": safe_get(data, "context_optimization.total_chunks_wasted"),
            # Meta
            "qa_pairs_evaluated": data.get("qa_pairs_evaluated"),
            "total_evaluation_time": data.get("total_evaluation_time"),
            "random_seed": data.get("random_seed"),
        }
        records.append(rec)

    return pd.DataFrame.from_records(records)


def main():
    ap = argparse.ArgumentParser(description="Aggregate evaluation_summary JSONs to a results table.")
    ap.add_argument("--in", dest="in_dir", type=Path, default=SUMMARY_DIR, help="Input dir (default: results/runs)")
    ap.add_argument("--out", dest="out_dir", type=Path, default=OUT_DIR, help="Output dir for tables (default: results/tables)")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(in_dir)

    # Sortierung: DSGVO-Score, ansonsten F1
    sort_col = "avg_dsgvo_score" if "avg_dsgvo_score" in df.columns and df["avg_dsgvo_score"].notna().any() else "avg_f1"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    csv_path = out_dir / "retrieval_results.csv"
    md_path  = out_dir / "retrieval_results.md"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_markdown(md_path, index=False)
    except Exception:
        # markdown ist optional (z. B. wenn tabulate nicht verfügbar ist)
        pass

    print(f"✅ Ergebnistabelle gespeichert: {csv_path}")
    if md_path.exists():
        print(f"📝 Markdown gespeichert:       {md_path}")
    print(df.head())


if __name__ == "__main__":
    main()
