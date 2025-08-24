# Research RAG

Ein modulates RAG-System, welches eine systematische Evaluation ermöglicht.

## Nutzung

1. Clone

2. Branch

3. Virtual Environment erstellen (.venv)

```bash
py -3.12 -m venv .venv
```

4. Virtual Environment aktivieren

```bash
./.venv/bin/activate
```

4. Requirements installieren

```bash
pip install -r requirements.txt
```

5. `.env` erstellen (an `.env.example`orientieren; OPENAI_API_KEY setzen)

6. Ausführen

```bash
 python src/rag/main.py --config configs/000_baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-20_1824.jsonl --num-qa 5
 ```

---

## Module

### Indexer

### Sprachmodell

Research-RAG ermöglicht die Einbindung verschiedener Sprachmodelle.

```bash
ollama pull mixtral:8x7b
ollama pull mixtral:8x22b
```


## Evaluation

### Ziel: zweistufige Evaluation des RAG-Systems.

1. Retrieval-Ebene (IR-Metriken): Grid-Search über Chunking/Parameter

2. Antwort-Ebene (RAGAS + DSGVO-Score): Vergleich von SLMs

### Voraussetzungen (Baseline)

In configs/000_baseline.yaml festlegen:

- dataset.path und dataset.evaluation_subset_size (für Smoke-Tests z. B. 1) 

- embedding.model_name und embedding.abbr (für Dateinamen) 

- LLM-Defaults (z. B. llm.model, llm.temperature, llm.max_tokens) 

- Kontextgrenze: pipeline.max_context_length (= max_content_size) 

### Retrieval-Grid (IR-Metriken)

Variierte Parameter:
- chunking.type
- chunking.chunk_size
- retrieval.top_k
- retrieval.similarity_threshold
- dataset.grouping.enabled.
Ergebnisse (Detail + Summary, mit Zeitstempel) landen in results/runs/. 

```bash
# Beispiel: alle Retrieval-Kombinationen ausführen
python src/rag/retrieval_grid_search.py
```

Hinweis: Das Skript liest die Baseline, überschreibt nur die o. g. Retrieval-Parameter und speichert pro Run JSON-Dateien (Zeitstempel kommt vom Evaluator). 

LLM-Grid (Antwort-Ebene)