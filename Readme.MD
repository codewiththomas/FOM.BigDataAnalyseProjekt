# Forschungs-RAG-System für Big Data Analyse

Ein modulares, erweiterbares RAG-System (Retrieval-Augmented Generation) für Forschungszwecke mit DSGVO-Daten.

## 🎯 Projektziel

Dieses System ermöglicht die systematische Evaluierung verschiedener RAG-Komponenten:

- **Chunker**: Line Chunker vs. Recursive Character Chunker
- **Embeddings**: OpenAI Embeddings vs. lokale Sentence Transformers
- **Vector Stores**: In-Memory vs. Chroma DB
- **Language Models**: OpenAI LLM vs. lokale Small Language Models (SLM)

## 📊 Evaluierungsmetriken

- **Precision, Recall, F1-Score**
- **RAGAS-Metriken** (Context Relevance, Answer Relevance, Faithfulness)
- **Mean Reciprocal Rank (MRR)**
- **Normalized Discounted Cumulative Gain (nDCG)**

## 🏗️ Systemarchitektur

```
src/
├── components/
│   ├── chunkers/           # Text-Aufteilung
│   │   ├── base_chunker.py
│   │   ├── line_chunker.py
│   │   └── recursive_chunker.py
│   ├── embeddings/         # Text-Vektorisierung
│   │   ├── base_embedding.py
│   │   ├── sentence_transformer_embedding.py
│   │   └── openai_embedding.py
│   ├── vector_stores/      # Vektorspeicherung
│   │   ├── base_vector_store.py
│   │   ├── in_memory_vector_store.py
│   │   └── chroma_vector_store.py
│   └── language_models/    # Antwortgenerierung
│       ├── base_language_model.py
│       ├── openai_language_model.py
│       └── local_language_model.py
├── config/                 # Konfiguration
├── evaluations/            # Evaluierung
├── experiments/            # Experimente
└── data_loader.py          # Datenverarbeitung
```

## 🚀 Schnellstart

### 1. Umgebung einrichten

```bash
# Repository klonen
git clone <repository-url>
cd FOM.BigDataAnalyseProjekt

# Python-Umgebung erstellen
python -m venv .venv

# Umgebung aktivieren (Windows)
.\.venv\Scripts\Activate

# Umgebung aktivieren (Linux/Mac)
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. OpenAI API Key konfigurieren

Erstellen Sie eine `.env`-Datei im Hauptverzeichnis:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Experimente ausführen

```bash
# Vollständige Experimente mit DSGVO-Daten
python run_experiments.py
```

### 4. Jupyter Notebook verwenden

```bash
# Jupyter starten
jupyter notebook src/rag.ipynb
```

## 📋 Verfügbare Komponenten

### Chunker

- **Line Chunker**: Einfache Zeilen-basierte Aufteilung
- **Recursive Character Chunker**: Intelligente Aufteilung mit Overlap

### Embeddings

- **Sentence Transformers**: Lokale Embeddings (all-MiniLM-L6-v2)
- **OpenAI Embeddings**: Cloud-basierte Embeddings (text-embedding-ada-002)

### Vector Stores

- **In-Memory**: Schneller Speicher für kleine Datensätze
- **Chroma DB**: Persistente, skalierbare Vektordatenbank

### Language Models

- **OpenAI GPT**: Cloud-basierte Antwortgenerierung
- **Lokale SLMs**: Ollama/LLaMA2 Integration

## 🔬 Experimente

Das System führt automatisch Experimente mit verschiedenen Konfigurationen durch:

1. **Baseline**: Line Chunker + Sentence Transformers + In-Memory + OpenAI
2. **Recursive Chunker**: Verbesserte Textaufteilung
3. **OpenAI Embeddings**: Cloud-basierte Vektorisierung
4. **Chroma Vector Store**: Persistente Speicherung
5. **Lokales LLM**: Offline-Verarbeitung
6. **Beste Kombination**: Optimierte Konfiguration

## 📊 Ergebnisse

Experimente werden automatisch gespeichert in:
- `experiment_results/` - JSON-Dateien mit detaillierten Ergebnissen
- `experiment_report.md` - Markdown-Bericht mit Vergleich

## 🛠️ Entwicklung

### Neue Komponenten hinzufügen

1. **Chunker**: Erweitern Sie `BaseChunker`
2. **Embeddings**: Erweitern Sie `BaseEmbedding`
3. **Vector Stores**: Erweitern Sie `BaseVectorStore`
4. **Language Models**: Erweitern Sie `BaseLanguageModel`

### Beispiel für neue Komponente

```python
from src.components.chunkers.base_chunker import BaseChunker

class MyCustomChunker(BaseChunker):
    def chunk_text(self, text: str) -> List[str]:
        # Ihre Implementierung hier
        pass
```

## 📈 Evaluierung

### Automatische Metriken

- **Retrieval-Metriken**: Precision@k, Recall@k, F1@k, MRR, nDCG
- **Generation-Metriken**: Relevanz, Treue, Korrektheit
- **System-Metriken**: Antwortlänge, Erfolgsrate, Fehlerrate

### Manuelle Evaluierung

```python
from src.evaluations.rag_metrics import RAGMetrics

metrics = RAGMetrics()
results = metrics.evaluate_rag_system(
    queries=test_questions,
    retrieved_docs=retrieved_documents,
    answers=generated_answers
)
```

## 📚 DSGVO-Daten

Das System verwendet die vollständige DSGVO-Verordnung als Testdaten:

- **Quelle**: `data/raw/dsgvo.txt`
- **Größe**: ~400KB, 3633 Zeilen
- **Struktur**: Artikel-basierte Aufteilung
- **Testfragen**: 20 DSGVO-spezifische Fragen

## 🔧 Konfiguration

### RAGConfig

```python
from src.config import RAGConfig

config = RAGConfig(
    chunker_type="recursive",
    chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
    embedding_type="openai",
    embedding_params={"model": "text-embedding-ada-002"},
    vector_store_type="chroma",
    vector_store_params={"persist_directory": "./chroma_db"},
    language_model_type="openai",
    language_model_params={"api_key": "your-key"}
)
```

## 🧪 Testing

```bash
# Unit Tests
pytest tests/unit/

# Integration Tests
pytest tests/integration/

# End-to-End Tests
pytest tests/e2e/
```

## 📖 Dokumentation

```bash
# Dokumentation generieren
mkdocs serve
```

## 🤝 Beitragen

1. Fork das Repository
2. Erstellen Sie einen Feature Branch
3. Implementieren Sie Ihre Änderungen
4. Fügen Sie Tests hinzu
5. Erstellen Sie einen Pull Request

## 📄 Lizenz

Dieses Projekt ist für Forschungszwecke konzipiert.

## 🆘 Support

Bei Fragen oder Problemen:
1. Überprüfen Sie die Dokumentation
2. Schauen Sie in die Issues
3. Erstellen Sie ein neues Issue mit detaillierter Beschreibung

---

**Entwickelt für das FOM Big Data Analyseprojekt** 🎓
