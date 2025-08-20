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