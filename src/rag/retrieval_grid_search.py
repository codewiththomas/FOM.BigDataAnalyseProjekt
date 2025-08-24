# src/rag/retrieval_grid_search.py
from __future__ import annotations

import os
import json
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Import aus deinem Projekt
from evaluator import RAGEvaluator  # unverändert verwenden

# --- Setup & Konstanten ------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../FOM.BigDataAnalyseProjekt
BASELINE_PATH = PROJECT_ROOT / "configs" / "000_baseline.yaml"
OUT_DIR = PROJECT_ROOT / "results" / "runs" / "retrieval"

# Grid NUR für Retrieval-Ebene (Embedding & LLM-Params bleiben fix)
PARAMETER_GRID = {
    "chunking_type": ["fixedsize", "recursive", "semantic"], # "fixedsize", "recursive", "semantic"
    "chunk_size": [1200, 1800, 2000],  # nur für fixedsize/recursive
    "top_k": [3, 5, 7], # 3, 5, 7
    "similarity_threshold": [0.00, 0.10],
    "grouping_enabled": [True, False], # True oder False oder True, False
}


def load_baseline_config() -> dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_embedding_abbr(cfg: dict) -> str:
    emb = cfg.get("embedding", {}) or {}
    if emb.get("abbr"):
        return str(emb["abbr"])
    # Fallback: heuristische Abkürzung
    name = (emb.get("model_name") or emb.get("model") or "").lower()
    if "all-mpnet-base-v2" in name:
        return "mpnet"
    if "all-minilm-l6-v2" in name:
        return "st-mini"
    return "emb"


def generate_experiment_name(
    chunking_type: str,
    chunk_size: Optional[int],
    top_k: int,
    similarity_threshold: float,
    emb_abbr: str,
) -> str:
    thr = f"s{int(round(similarity_threshold * 100)):02d}"
    if chunking_type == "semantic":
        return f"000_baseline_semantic_k{top_k}_{thr}_{emb_abbr}"
    return f"000_baseline_{chunking_type}_{chunk_size}_k{top_k}_{thr}_{emb_abbr}"


def build_run_config(
    base_cfg: dict,
    chunking_type: str,
    chunk_size: Optional[int],
    top_k: int,
    similarity_threshold: float,
    grouping_enabled: bool,
) -> dict:
    """Erzeugt eine tiefe Kopie der Baseline und überschreibt nur die variierenden Felder."""
    cfg = deepcopy(base_cfg)

    # Chunking
    ch = cfg.setdefault("chunking", {})
    ch["type"] = chunking_type
    if chunking_type in ("fixedsize", "recursive"):
        ch["chunk_size"] = int(chunk_size)
        # >>> 15% Overlap aus chunk_size berechnen
        overlap = int(round(ch["chunk_size"] * 0.15))
        # zur Sicherheit clampen (Overlap < chunk_size, >= 0)
        overlap = max(0, min(overlap, ch["chunk_size"] - 1))
        ch["chunk_overlap"] = overlap
    else:
        ch.pop("chunk_size", None)  # semantic ignoriert chunk_size
        ch.pop("chunk_overlap", None)

    # Retrieval
    ret = cfg.setdefault("retrieval", {})
    ret["top_k"] = int(top_k)
    ret["similarity_threshold"] = float(similarity_threshold)

    # Grouping Toggle
    ds = cfg.setdefault("dataset", {})
    grp = ds.setdefault("grouping", {})
    grp["enabled"] = bool(grouping_enabled)

    return cfg


@contextmanager
def pushd(new_dir: Path):
    prev = Path.cwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)


def main():
    load_dotenv()
    print("✅ .env file loaded in grid search")

    base_cfg = load_baseline_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output (retrieval runs): {OUT_DIR.resolve()}")

    # Informationen aus Baseline (Embedding bleibt fix; Abkürzung kommt aus Config)
    emb_abbr = get_embedding_abbr(base_cfg)
    dataset_path = Path(base_cfg["dataset"]["path"]).resolve()
    num_qa = int(base_cfg.get("dataset", {}).get("evaluation_subset_size", 100))

    # Kombis bauen (chunk_size nur für fixedsize/recursive)
    combos = []
    for ctype in PARAMETER_GRID["chunking_type"]:
        if ctype in ("fixedsize", "recursive"):
            iterable = product(
                PARAMETER_GRID["chunk_size"],
                PARAMETER_GRID["top_k"],
                PARAMETER_GRID["similarity_threshold"],
                PARAMETER_GRID["grouping_enabled"],
            )
            for cs, k, thr, g in iterable:
                combos.append((ctype, cs, k, thr, g))
        else:
            iterable = product(
                PARAMETER_GRID["top_k"],
                PARAMETER_GRID["similarity_threshold"],
                PARAMETER_GRID["grouping_enabled"],
            )
            for k, thr, g in iterable:
                combos.append((ctype, None, k, thr, g))

    print(f"🔍 Grid Search: {len(combos)} Kombinationen")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🔢 QA Paare pro Run: {num_qa}")
    print("------------------------------------------------------------\n")

    # In den Ergebnisordner wechseln, damit Evaluator alle JSONs hier ablegt
    with pushd(OUT_DIR):
        for idx, (ctype, cs, k, thr, g_enabled) in enumerate(combos, start=1):
            exp_name = generate_experiment_name(ctype, cs, k, thr, emb_abbr)
            print(f"📋 {idx}/{len(combos)}: {exp_name}")

            # Lauf-spezifische Config erzeugen (Temp-YAML)
            run_cfg = build_run_config(base_cfg, ctype, cs, k, thr, g_enabled)
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tf:
                yaml.safe_dump(run_cfg, tf, sort_keys=False, allow_unicode=True)
                temp_config_path = Path(tf.name)

            try:
                evaluator = RAGEvaluator(
                    config_path=str(temp_config_path), dataset_path=str(dataset_path)
                )
                # Falls der Evaluator den Namen aus der Config liest, ist das optional.
                # Setzen schadet nicht:
                evaluator.experiment_name = exp_name

                # Speicher alle Metriken (inkl. DSGVO-Score) – wir filtern NICHTS heraus
                _summary = evaluator.run_evaluation(
                    num_qa=num_qa, save_results=True
                )

                # Optionales Cache-Aufräumen pro Run (deine Vorgabe)
                try:
                    evaluator.clear_cache()
                except Exception:
                    pass

            except Exception as e:
                print(f"❌ {exp_name} fehlgeschlagen")
                print(f"   Error: {e}")
            finally:
                # Temp-YAML löschen
                try:
                    temp_config_path.unlink(missing_ok=True)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
