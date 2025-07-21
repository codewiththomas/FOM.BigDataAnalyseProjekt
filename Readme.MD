# Simple Research RAG 🔍

Ein einfaches, modulares RAG-System (Retrieval-Augmented Generation) zur Evaluierung verschiedener Sprachmodelle, mit besonderem Fokus auf deutsche und On-Premise-Modelle.

## 🎯 Projektziel

Dieses Projekt ermöglicht die systematische Evaluierung und den Vergleich verschiedener Sprachmodelle für deutsche Dokumente durch ein benutzerfreundliches RAG-System.

## ✨ Features

- **Modulare Architektur**: Einfacher Wechsel zwischen verschiedenen Sprachmodellen
- **Mehrere Model-Provider**: OpenAI, Groq, Ollama (lokal)
- **Umfassende Evaluierung**: RAGAS, BERTScore, Performance-Metriken
- **Web-Interface**: Streamlit-basierte Benutzeroberfläche
- **Deutsche Optimierung**: Spezielle Unterstützung für deutsche Texte

## 🏗️ Architektur

```
├── src/
│   ├── data_loader.py      # Dokumenten-Loading und Chunking
│   ├── retriever.py        # FAISS-basierte Vektorsuche
│   ├── generator.py        # LLM-Provider (OpenAI, Groq, Ollama)
│   ├── evaluator.py        # Evaluierungsmetriken
│   └── main.py            # Hauptorchestrator
├── data/
│   ├── documents/          # Dokumente für RAG
│   └── test_questions.json # Testfragen mit Referenzantworten
├── results/               # Evaluierungsergebnisse
├── config.yaml           # Systemkonfiguration
└── app.py                # Streamlit Web-Interface
```

## 🚀 Installation

### 1. Repository klonen
```bash
git clone <repository-url>
cd ResearchRAG
```

### 2. Virtuelle Umgebung erstellen
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Umgebungsvariablen setzen
```bash
# .env Datei erstellen oder direkt setzen
export OPENAI_API_KEY="your-openai-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

### 5. Lokale Modelle installieren (optional)
Für Ollama-Modelle:
```bash
# Ollama installieren: https://ollama.ai/
ollama pull llama3.1:8b-sauerkraut
ollama pull qwen2.5:7b-instruct
```

## 📖 Nutzung

### Command Line Interface

#### 1. System Setup
```bash
# Daten laden und Index erstellen
python src/main.py --setup

# Index neu erstellen (force)
python src/main.py --rebuild
```

#### 2. Einzelne Anfrage
```bash
python src/main.py --query "Was ist die DSGVO?"
```

#### 3. Modell wechseln
```bash
# Verfügbare Modelle anzeigen
python src/main.py --list-models

# Zu anderem Modell wechseln
python src/main.py --model gpt-4o-mini
python src/main.py --model mixtral-8x7b
python src/main.py --model llama-sauerkraut
```

#### 4. Evaluierung ausführen
```bash
# Vollständige Evaluierung
python src/main.py --evaluate

# Begrenzte Testgröße
python src/main.py --evaluate --test-size 10
```

#### 5. Modelle vergleichen
```bash
python src/main.py --compare
```

### Web Interface

```bash
streamlit run app.py
```

Das Web-Interface bietet:
- **Query Tab**: Interaktive Fragestellung mit Kontextanzeige
- **Evaluation Tab**: Modell-Evaluierung mit Metriken
- **Results Analysis**: Vergleich verschiedener Modelle
- **Settings**: Systemkonfiguration und Diagnostik

## 🔧 Konfiguration

Die `config.yaml` Datei steuert alle Systemparameter:

```yaml
# Modell-Konfiguration
model:
  name: "gpt-4o-mini"
  provider: "openai"
  temperature: 0.1
  max_tokens: 1000

# Retrieval-Konfiguration
retrieval:
  embedding_model: "text-embedding-3-small"
  chunk_size: 500
  chunk_overlap: 100
  top_k: 5

# Evaluierungs-Konfiguration
evaluation:
  metrics: ["ragas", "bertscore", "performance"]
  test_size: 50
```

## 🤖 Unterstützte Modelle

| Modell | Provider | Beschreibung |
|--------|----------|--------------|
| `gpt-4o-mini` | OpenAI | Schnell und kostengünstig |
| `mixtral-8x7b` | Groq | Mixtral 8x7B via Groq API |
| `llama-sauerkraut` | Ollama | Deutsch-optimiertes Llama |
| `qwen2.5-7b` | Ollama | Qwen 2.5 7B Instruct |

## 📊 Evaluierungsmetriken

### RAGAS Metriken
- **Answer Relevancy**: Relevanz der Antwort zur Frage
- **Faithfulness**: Treue zum Kontext
- **Context Relevancy**: Relevanz des abgerufenen Kontexts

### BERTScore
- **Precision, Recall, F1**: Semantische Ähnlichkeit mit Referenzantworten

### Performance Metriken
- **Execution Time**: Antwortzeit
- **Tokens/Second**: Durchsatz
- **Cost Estimate**: Geschätzte Kosten (API-Modelle)

## 📁 Datenstruktur

### Dokumente hinzufügen
Legen Sie Textdateien (.txt, .md) in `data/documents/` ab:
```
data/documents/
├── dsgvo_beispiel.txt
├── weitere_dokumente.txt
└── ...
```

### Testfragen erstellen
Format für `data/test_questions.json`:
```json
[
  {
    "id": 1,
    "question": "Was ist die DSGVO?",
    "reference_answer": "Die DSGVO ist eine EU-Verordnung...",
    "category": "basics"
  }
]
```

## 🔍 Beispiel-Workflow

1. **Setup**:
   ```bash
   python src/main.py --setup
   ```

2. **Baseline erstellen**:
   ```bash
   python src/main.py --model gpt-4o-mini
   python src/main.py --evaluate --test-size 20
   ```

3. **Alternativen testen**:
   ```bash
   python src/main.py --model mixtral-8x7b
   python src/main.py --evaluate --test-size 20

   python src/main.py --model llama-sauerkraut
   python src/main.py --evaluate --test-size 20
   ```

4. **Ergebnisse vergleichen**:
   ```bash
   python src/main.py --compare
   ```

## 📈 Ergebnisse

Evaluierungsergebnisse werden in `results/` gespeichert:
- `{model}_detailed_{timestamp}.json`: Detailierte Ergebnisse
- `{model}_aggregates_{timestamp}.json`: Zusammengefasste Metriken
- `{model}_results_{timestamp}.csv`: CSV für weitere Analyse
- `model_comparison.csv`: Modellvergleich

## 🔧 Troubleshooting

### Häufige Probleme

1. **API-Schlüssel fehlen**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export GROQ_API_KEY="your-key"
   ```

2. **Ollama nicht erreichbar**:
   ```bash
   ollama serve  # Ollama-Server starten
   ```

3. **Dependencies fehlen**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Speicher-Probleme bei FAISS**:
   - Chunk-Größe reduzieren in `config.yaml`
   - Weniger Dokumente verwenden

### Diagnostik ausführen
```bash
# CLI
python src/main.py --help

# Web-Interface
# Gehen Sie zu Settings > System Diagnostics
```

## 🤝 Entwicklung

### Neue Modelle hinzufügen

1. **Generator erweitern** (`src/generator.py`):
   ```python
   class NewModelGenerator(BaseGenerator):
       def generate(self, query, context):
           # Implementation
   ```

2. **Factory-Funktion aktualisieren**:
   ```python
   def get_generator(config):
       if config["provider"] == "new_provider":
           return NewModelGenerator(config)
   ```

3. **Modell zu `AVAILABLE_MODELS` hinzufügen**

### Neue Metriken hinzufügen

1. **Evaluator erweitern** (`src/evaluator.py`):
   ```python
   def _evaluate_custom_metric(self, generated, reference):
       # Implementation
   ```

2. **Metrik in `evaluate_single` integrieren**

## 📝 Lizenz

MIT License - siehe LICENSE Datei für Details.

## 🙋‍♂️ Support

Bei Fragen oder Problemen:
1. Überprüfen Sie die Troubleshooting-Sektion
2. Führen Sie die Systemdiagnose aus
3. Öffnen Sie ein Issue im Repository

## 🔮 Roadmap

- [ ] Weitere Embedding-Modelle
- [ ] Advanced Chunking-Strategien
- [ ] Multi-Modal RAG
- [ ] Erweiterte Evaluierungsmetriken
- [ ] Docker-Container
- [ ] Cloud-Deployment

---

**Happy RAG-ing! 🚀**