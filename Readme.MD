# ResearchRAG - Modulares RAG-System

Ein flexibles und erweiterbares RAG-System für wissenschaftliche Studien, das es ermöglicht, verschiedene Komponenten (Chunker, Embeddings, Vector Stores, Language Models) einfach auszutauschen und zu vergleichen.

## Projektübersicht

Dieses Projekt wurde für eine wissenschaftliche Studie (Hausarbeit; 4 Personen; 30 Tage) entwickelt, bei der jede Person einen spezifischen Aspekt des RAG-Systems untersucht:

- **Person A**: Chunking-Strategien (verschiedene Chunker-Implementierungen)
- **Person B**: Embedding-Verfahren (verschiedene Embedding-Modelle)
- **Person C**: Vector Stores & Retrieval (verschiedene Speicher- und Suchstrategien)
- **Person D**: Language Models & Generation (verschiedene LLM-Ansätze)

## Architektur

### Modularer Aufbau
```
src/
├── components/           # Austauschbare RAG-Komponenten
│   ├── chunkers/        # Text-Chunking-Strategien
│   │   ├── base_chunker.py
│   │   ├── line_chunker.py
│   │   ├── recursive_chunker.py
│   │   └── semantic_chunker.py
│   ├── embeddings/      # Embedding-Modelle
│   │   ├── base_embedding.py
│   │   ├── openai_embedding.py
│   │   └── sentence_transformer_embedding.py
│   ├── vector_stores/   # Vector-Speicher-Systeme
│   │   ├── base_vector_store.py
│   │   ├── in_memory_vector_store.py
│   │   ├── chroma_vector_store.py
│   │   └── faiss_vector_store.py
│   └── language_models/ # Language Models
│       ├── base_language_model.py
│       ├── openai_language_model.py
│       └── ollama_language_model.py
├── config/              # Konfigurationssystem
│   ├── base_config.py
│   ├── pipeline_configs.py
│   └── experiment_configs.py
├── core/                # RAG-Pipeline und Component Loader
│   ├── rag_pipeline.py
│   ├── component_loader.py
│   └── experiment_runner.py
├── evaluations/         # Evaluierungsmetriken
│   ├── base_evaluator.py
│   ├── retrieval_evaluator.py
│   ├── generation_evaluator.py
│   ├── rag_evaluator.py
│   └── performance_evaluator.py
├── utils/               # Hilfsfunktionen
│   └── data_loader.py
├── ResearchRAG.ipynb    # Hauptnotebook
└── test_evaluation_system.py
```

### Datenstruktur
```
data/
├── raw/                 # Rohdaten (DSGVO-Text)
│   ├── dsgvo.txt
│   └── dsgvo_timo.txt
├── processed/           # Verarbeitete Daten
│   ├── chunks/         # Gespeicherte Chunks
│   └── embeddings/     # Gespeicherte Embeddings
└── evaluation/         # Evaluierungsdaten
    ├── results/        # Experiment-Ergebnisse
    └── qa_pairs.json   # QA-Datensatz
```

## Schnellstart

### 1. Installation

```bash
# Repository klonen
git clone <repository-url>
cd FOM.BigDataAnalyseProjekt

# Abhängigkeiten installieren
pip install -r requirements.txt

# Paket im Entwicklungsmodus installieren (empfohlen)
pip install -e .
```

**Ohne Installation (Portabel):**
```bash
# Projekt in beliebiges Verzeichnis kopieren
cp -r FOM.BigDataAnalyseProjekt /mycode
cd /mycode

# Abhängigkeiten installieren
pip install -r requirements.txt

# Notebooks funktionieren automatisch mit robuster Import-Lösung
```

### 2. Umgebung konfigurieren

```bash
# API-Schlüssel setzen
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Jupyter Notebook starten

```bash
# Notebook öffnen
jupyter notebook src/ResearchRAG.ipynb
```

Das Notebook führt Sie durch:
- Pipeline-Konfiguration
- Dokument-Indexierung
- Query-Ausführung
- Komponenten-Analyse

## Verfügbare Komponenten

### Implementierte Komponenten

**Chunker:**
- `LineChunker` - Einfaches zeilenbasiertes Chunking
- `RecursiveChunker` - Rekursives Text-Chunking mit Overlap
- `SemanticChunker` - Semantisches Chunking basierend auf Bedeutung

**Embeddings:**
- `OpenAIEmbedding` - OpenAI text-embedding-3-small
- `SentenceTransformerEmbedding` - Sentence Transformers Modelle

**Vector Stores:**
- `InMemoryVectorStore` - In-Memory-Speicher mit sklearn
- `ChromaVectorStore` - ChromaDB Integration
- `FAISSVectorStore` - FAISS Integration

**Language Models:**
- `OpenAILanguageModel` - OpenAI GPT-Modelle
- `OllamaLanguageModel` - Lokale Ollama-Modelle

### Baseline-Konfiguration

```python
baseline_config = {
    "chunker": {
        "type": "line_chunker",
        "chunk_size": 512,
        "overlap": 50
    },
    "embedding": {
        "type": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536
    },
    "vector_store": {
        "type": "in_memory",
        "similarity_metric": "cosine",
        "top_k": 5
    },
    "language_model": {
        "type": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 500
    }
}
```

## Beispiel-Verwendung

```python
from src.config.pipeline_configs import get_baseline_config
from src.core.rag_pipeline import RAGPipeline

# Baseline-Konfiguration laden
config = get_baseline_config()

# Pipeline erstellen
pipeline = RAGPipeline(config)

# Dokumente indexieren
documents = pipeline.load_documents_from_file("data/raw/dsgvo.txt")
pipeline.index_documents(documents)

# Query ausführen
result = pipeline.query("Was ist die maximale Geldbuße nach Art. 83 DSGVO?")
print(result['answer'])
```

## Experimentieren

### Neue Komponente hinzufügen

1. **Basisklasse implementieren**:
```python
from src.components.chunkers import BaseChunker

class MyCustomChunker(BaseChunker):
    def chunk_text(self, text: str) -> List[str]:
        # Ihre Implementierung hier
        pass
```

2. **Komponente registrieren**:
```python
from src.core.component_loader import component_loader
component_loader.register_chunker("my_custom", MyCustomChunker)
```

3. **Konfiguration erstellen**:
```python
from src.config.pipeline_configs import create_custom_config

config = create_custom_config(
    chunker_type="my_custom",
    embedding_type="openai",
    vector_store_type="in_memory",
    llm_type="openai"
)
```

### Konfigurationen vergleichen

```python
from src.config.pipeline_configs import get_alternative_configs

configs = get_alternative_configs()
for name, config in configs.items():
    print(f"{name}: {config.get_component_types()}")
```

## Evaluierung

### QA-Datensatz
Das Projekt enthält einen DSGVO-spezifischen QA-Datensatz mit verschiedenen Fragetypen:
- **Faktenfragen**: "Was ist die maximale Geldbuße?"
- **Prozessfragen**: "Wie läuft eine Datenschutz-Folgenabschätzung ab?"
- **Problemlösungsfragen**: "Was ist bei Marketing-Nutzung zu beachten?"

### Verfügbare Evaluierungsmetriken

**Retrieval-Metriken:**
- Precision@k, Recall@k, F1@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

**Generation-Metriken:**
- ROUGE-L, BLEU
- BERTScore
- Semantic Similarity

**RAG-Metriken:**
- RAGAS Framework
- Faithfulness (Hallucination Rate)
- Groundedness

**Performance-Metriken:**
- Inferenzzeit
- Speicherverbrauch
- Durchsatz

### Experiment-Runner verwenden

```python
from src.core.experiment_runner import ExperimentRunner

# Experiment-Runner erstellen
runner = ExperimentRunner()

# Einzelnes Experiment ausführen
results = runner.run_experiment(config, qa_pairs)

# Mehrere Konfigurationen vergleichen
comparison = runner.compare_configurations([config1, config2, config3])
```

## Entwicklung

### Tests ausführen

```bash
# Alle Tests ausführen
pytest tests/

# Spezifische Tests
python run_tests.py

# Evaluierungssystem testen
python src/test_evaluation_system.py
```

### Code-Qualität

```bash
# Code formatieren
black src/

# Linting
flake8 src/
```

## Für Google Colab

Das System ist für Google Colab optimiert:

1. **Automatische Package-Installation**
2. **Google Drive-Integration** für Persistierung
3. **GPU-Unterstützung** für Embeddings
4. **Memory-Management** für große Dokumente

```python
# In Google Colab
!git clone <repository-url>
%cd FOM.BigDataAnalyseProjekt
!pip install -r requirements.txt

# Google Drive mounten
from google.colab import drive
drive.mount('/content/drive')
```

## Beitragen

### Für Teammitglieder

1. **Fork** das Repository
2. **Branch** für Ihr Feature erstellen (`git checkout -b feature/chunking-strategy`)
3. **Commit** Ihre Änderungen (`git commit -am 'Add new chunking strategy'`)
4. **Push** zum Branch (`git push origin feature/chunking-strategy`)
5. **Pull Request** erstellen

### Coding-Standards

- **Docstrings**: Alle öffentlichen Funktionen dokumentieren
- **Type Hints**: Verwenden Sie Type Annotations
- **Tests**: Mindestens 80% Code Coverage
- **Logging**: Verwenden Sie das logging-Modul

## Lizenz

Dieses Projekt ist für Bildungszwecke an der FOM entwickelt.

## Team

- **Person A**: Chunking-Strategien
- **Person B**: Embedding-Verfahren
- **Person C**: Vector Stores & Retrieval
- **Person D**: Language Models & Generation

## Nützliche Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://faiss.ai/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)

---

**Status**: Vollständig implementiert (Alle Komponenten verfügbar)
**Nächste Schritte**: Experimentelle Evaluierung und Optimierung
