# 🚀 ResearchRAG - Lokale Nutzung

Vollständig lokales RAG-System ohne API-Abhängigkeiten für die DSGVO-Analyse.

## ⚡ Schnellstart

### 1. Automatisches Setup (Windows)
```bash
setup_local.bat
```

### 2. Manuelles Setup

**Dependencies installieren:**
```bash
pip install sentence-transformers chromadb faiss-cpu requests
```

**Ollama installieren:**
- Windows: https://ollama.ai/download
- Linux/Mac: `curl -fsSL https://ollama.ai/install.sh | sh`

**Ollama starten und Modell laden:**
```bash
ollama serve
ollama pull llama3.2
```

### 3. System testen
```bash
python test_local_rag.py
```

## 🔧 Konfiguration

Das System nutzt vollständig lokale Komponenten:

- **Chunker**: RecursiveChunker (1000 Zeichen, 200 Überlapp)
- **Embedding**: SentenceTransformer (all-MiniLM-L6-v2, 384D)
- **Vector Store**: InMemory (Cosine-Ähnlichkeit)
- **LLM**: Ollama (llama3.2)

## 📋 Verwendung

### Basis-Pipeline
```python
from config.pipeline_configs import get_local_config
from core.rag_pipeline import RAGPipeline

# Lokale Konfiguration
config = get_local_config()
pipeline = RAGPipeline(config)

# Dokument indexieren
documents = pipeline.load_documents_from_file("data/raw/dsgvo.txt")
pipeline.index_documents(documents)

# Frage stellen
result = pipeline.query("Was ist die maximale Geldbuße nach Art. 83 DSGVO?")
print(result["answer"])
```

### QA-Evaluierung
```python
import json

# QA-Pairs laden
with open("data/evaluation/qa_pairs.json", "r") as f:
    qa_data = json.load(f)

# Fragen testen
for qa in qa_data["questions"][:5]:
    result = pipeline.query(qa["question"])
    print(f"Frage: {qa['question']}")
    print(f"Antwort: {result['answer']}")
    print(f"Gold: {qa['gold_answer']}")
    print("-" * 50)
```

## 🔍 Verfügbare Komponenten

### Chunker
- `line_chunker`: Einfaches zeilenbasiertes Chunking
- `recursive_chunker`: Hierarchisches Chunking mit Separatoren
- `semantic_chunker`: Semantisches Chunking basierend auf Bedeutung

### Embeddings
- `sentence_transformer`: Lokale SentenceTransformer-Modelle
  - `all-MiniLM-L6-v2` (384D, schnell)
  - `all-mpnet-base-v2` (768D, besser)

### Vector Stores
- `in_memory`: Schneller In-Memory-Speicher
- `chroma`: Persistente ChromaDB
- `faiss`: Hochperformante FAISS-Suche

### Language Models
- `ollama`: Lokale Ollama-Modelle
  - `llama3.2` (empfohlen)
  - `mistral`
  - `codellama`

## 🛠️ Anpassungen

### Eigene Konfiguration
```python
from config.pipeline_configs import PipelineConfig

custom_config = PipelineConfig({
    "chunker": {
        "type": "semantic_chunker",
        "chunk_size": 800,
        "overlap": 100
    },
    "embedding": {
        "type": "sentence_transformer",
        "model": "all-mpnet-base-v2",  # Bessere Qualität
        "normalize_embeddings": True
    },
    "language_model": {
        "type": "ollama",
        "model": "mistral",  # Anderes Modell
        "temperature": 0.0   # Deterministischer
    }
})
```

### Neues Modell hinzufügen
```bash
ollama pull mistral
ollama pull codellama
```

## 📊 Performance

**Typische Werte (lokaler Laptop):**
- Indexierung: ~2-5 Sekunden für DSGVO-Dokument
- Query-Zeit: ~1-3 Sekunden pro Frage
- Speicher: ~2-4 GB RAM
- Embedding-Dimension: 384 (MiniLM) oder 768 (mpnet)

## 🐛 Troubleshooting

### Ollama-Probleme
```bash
# Prüfe ob Ollama läuft
curl http://localhost:11434/api/tags

# Starte Ollama neu
ollama serve

# Prüfe verfügbare Modelle
ollama list
```

### Import-Fehler
```bash
# Installiere fehlende Dependencies
pip install sentence-transformers
pip install chromadb
pip install faiss-cpu
```

### Speicher-Probleme
- Nutze kleinere Chunk-Größen (512 statt 1000)
- Verwende `all-MiniLM-L6-v2` statt `all-mpnet-base-v2`
- Reduziere `batch_size` in der Embedding-Konfiguration

## ✅ System-Check

Führe aus: `python test_local_rag.py`

Das Skript prüft:
- ✅ Dependencies verfügbar
- ✅ Ollama läuft
- ✅ llama3.2 geladen
- ✅ Pipeline funktioniert
- ✅ QA-Evaluierung läuft

## 🎯 Nächste Schritte

1. **Experimentiere** mit verschiedenen Konfigurationen
2. **Evaluiere** mit dem vollständigen QA-Datensatz
3. **Optimiere** Performance für Deine Hardware
4. **Erweitere** um eigene Dokumente und Fragen