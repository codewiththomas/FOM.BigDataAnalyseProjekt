# ResearchRAG - Modulares RAG-System

Ein flexibles und erweiterbares RAG-System für wissenschaftliche Studien, das es ermöglicht, verschiedene Komponenten (Chunker, Embeddings, Vector Stores, Language Models) einfach auszutauschen und zu vergleichen.

## 🎯 Projektübersicht

Dieses Projekt wurde für eine wissenschaftliche Studie (Hausarbeit; 4 Personen; 30 Tage) entwickelt, bei der jede Person einen spezifischen Aspekt des RAG-Systems untersucht:

- **Person A**: Chunking-Strategien (verschiedene Chunker-Implementierungen)
- **Person B**: Embedding-Verfahren (verschiedene Embedding-Modelle)
- **Person C**: Vector Stores & Retrieval (verschiedene Speicher- und Suchstrategien)
- **Person D**: Language Models & Generation (verschiedene LLM-Ansätze)

## 🏗️ Architektur

### Modularer Aufbau
```
src/
├── components/           # Austauschbare RAG-Komponenten
│   ├── chunkers/        # Text-Chunking-Strategien
│   ├── embeddings/      # Embedding-Modelle
│   ├── vector_stores/   # Vector-Speicher-Systeme
│   └── language_models/ # Language Models
├── config/              # Konfigurationssystem
├── core/                # RAG-Pipeline und Component Loader
├── evaluations/         # Evaluierungsmetriken
└── utils/               # Hilfsfunktionen
```

### Datenstruktur
```
data/
├── raw/                 # Rohdaten (DSGVO-Text)
├── processed/           # Verarbeitete Daten
│   ├── chunks/         # Gespeicherte Chunks
│   └── embeddings/     # Gespeicherte Embeddings
└── evaluation/         # Evaluierungsdaten
    ├── results/        # Experiment-Ergebnisse
    └── qa_pairs.json   # QA-Datensatz
```

## 🚀 Schnellstart

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

## 🔧 Verfügbare Komponenten

### Baseline-Implementierungen (Phase 1)
- **Chunker**: `LineChunker` - Einfaches zeilenbasiertes Chunking
- **Embedding**: `OpenAIEmbedding` - OpenAI text-embedding-3-small
- **Vector Store**: `InMemoryVectorStore` - In-Memory-Speicher mit sklearn
- **Language Model**: `OpenAILanguageModel` - OpenAI GPT-4o-mini

### Geplante Erweiterungen (Phase 2-4)
- **Chunker**: `RecursiveChunker`, `SemanticChunker`
- **Embedding**: `SentenceTransformerEmbedding`, `HuggingFaceEmbedding`
- **Vector Store**: `ChromaVectorStore`, `FAISSVectorStore`
- **Language Model**: `OllamaLanguageModel`, `HuggingFaceLanguageModel`

## 📊 Beispiel-Konfiguration

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

## 🔬 Experimentieren

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

## 📈 Evaluierung

### QA-Datensatz
Das Projekt enthält einen DSGVO-spezifischen QA-Datensatz mit verschiedenen Fragetypen:
- **Faktenfragen**: "Was ist die maximale Geldbuße?"
- **Prozessfragen**: "Wie läuft eine Datenschutz-Folgenabschätzung ab?"
- **Problemlösungsfragen**: "Was ist bei Marketing-Nutzung zu beachten?"

### Metriken (geplant)
- **Retrieval**: Precision@k, Recall@k, F1@k, MRR, NDCG
- **Generation**: ROUGE-L, BLEU, BERTScore, Semantic Similarity
- **RAG**: RAGAS, Faithfulness, Groundedness
- **Performance**: Inferenzzeit, Speicherverbrauch, Durchsatz

## 🛠️ Entwicklung

### Projektstruktur erweitern

```bash
# Neue Komponente hinzufügen
touch src/components/chunkers/my_new_chunker.py

# Tests schreiben
touch tests/unit/test_my_new_chunker.py

# Dokumentation aktualisieren
echo "## MyNewChunker" >> docs/components.md
```

### Code-Qualität

```bash
# Tests ausführen
pytest tests/

# Code formatieren
black src/

# Linting
flake8 src/
```

## 📚 Für Google Colab

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

## 🤝 Beitragen

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

## 📄 Lizenz

Dieses Projekt ist für Bildungszwecke an der FOM entwickelt.

## 👥 Team

- **Person A**: Chunking-Strategien
- **Person B**: Embedding-Verfahren
- **Person C**: Vector Stores & Retrieval
- **Person D**: Language Models & Generation

## 🔗 Nützliche Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://faiss.ai/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)

---

**Status**: Phase 1 abgeschlossen (Baseline-System funktionsfähig)
**Nächste Schritte**: Implementierung erweiterte Komponenten (Phase 2-4)