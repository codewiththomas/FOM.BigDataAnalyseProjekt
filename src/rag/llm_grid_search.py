# src/rag/llm_grid_search.py
from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv

from evaluator import RAGEvaluator  # bestehend

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "runs" / "llm"

# Trage hier deine zwei Baselines ein (Dummy jetzt ok; später feintunen)
BASELINE_PATHS = [
    PROJECT_ROOT / "configs" / "000_baseline-conservative.yaml",
    PROJECT_ROOT / "configs" / "000_baseline-extended.yaml",
]

# Beispiel-Modelle (lokal/lite). Passe an deine Umgebung an.
LLM_MODELS = [
    # {"name": "llama3:8b", "abbr": "l3-8b"},
    # {"name": "qwen3:8b", "abbr": "q3-8b"},
    {"name": "qwen3:0.6b",  "abbr": "q3.06b"},
    # {"name": "mistral-nemo:12b", "abbr": "mn-12b"},
]

# Für Dummy Test: oberen Block auskommentieren, folgenden Block nutzen
# LLM_MODELS = [
#     {"name": "ignored"}  # Modell wird NICHT überschrieben; YAML llm.abbr wird für den Namen verwendet
# ]

# OPTIONAL: Temperatur-Studie (leer lassen = deaktiviert)
OPTIONAL_TEMPS = []  # z.B. [0.0, 0.1, 0.2]

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_temp_yaml(cfg: dict) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False, allow_unicode=True)
        return Path(tf.name)

@contextmanager
def pushd(new_dir: Path):
    prev = Path.cwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

def build_cfg(base: dict, model_name: str, model_abbr: str | None = None, temperature: float | None = None) -> dict:
    cfg = deepcopy(base)
    llm = cfg.setdefault("llm", {})
    # Modell ggf. überschreiben (wenn du die Zeile auskommentierst, bleibt das YAML-Modell)
    llm["model"] = model_name
    if model_abbr:
        llm["abbr"] = model_abbr
    if temperature is not None:
        llm["temperature"] = float(temperature)
    return cfg

def exp_name(base_label: str, model_abbr: str) -> str:
    # klare Benennung, getrennt von Retrieval-Runs
    return f"001_llm_{model_abbr}_{base_label}"

def main():
    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output (LLM runs): {OUT_DIR.resolve()}")

    for base_path in BASELINE_PATHS:
        base_cfg = load_yaml(base_path)
        base_label = base_path.stem.replace("-", "_")
        dataset_path = Path(base_cfg["dataset"]["path"]).resolve()  # <- jetzt korrekt
        num_qa = int(base_cfg["dataset"]["evaluation_subset_size"])

        print(f"\n🔧 Baseline: {base_path.name}")
        print(f"📊 Dataset:  {dataset_path}")
        print(f"🔢 QA Paare: {num_qa}")
        print("------------------------------------------------------------")

        base_llm_abbr = base_cfg.get("llm", {}).get("abbr", "llm")

        with pushd(OUT_DIR):
            for m in LLM_MODELS:
                model_name = m["name"]
                model_abbr = m.get("abbr")  # <- optional

                temps = OPTIONAL_TEMPS or [base_cfg.get("llm", {}).get("temperature", 0.1)]
                for temp in temps:
                    t_suffix = "" if not OPTIONAL_TEMPS else f"_t{int(round(float(temp) * 100)):02d}"
                    name_abbr = model_abbr or base_llm_abbr
                    name = exp_name(base_label, name_abbr) + t_suffix
                    print(f"▶️  {name}  (model={model_name}, temp={temp})")

                    if list(Path(".").glob(f"{name}_*summary_*.json")):
                        print(f"↩️  Skip {name} (bereits vorhanden)")
                        continue

                    tmp = save_temp_yaml(
                        build_cfg(base_cfg, model_name, model_abbr,
                                  None if not OPTIONAL_TEMPS else float(temp))
                    )
                    try:
                        ev = RAGEvaluator(config_path=str(tmp), dataset_path=str(dataset_path))
                        ev.experiment_name = name
                        _summary = ev.run_evaluation(num_qa=num_qa, save_results=True)
                        try:
                            ev.clear_cache()
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"❌ {name} fehlgeschlagen\n   Error: {e}")
                    finally:
                        try:
                            tmp.unlink(missing_ok=True)
                        except Exception:
                            pass

if __name__ == "__main__":
    main()
