# Chunking

## Übersicht

Mit dem Chunking-Modul werden Dokumente in kleinere, verarbeitbare Segmente (Chunks) unterteilt.

## Architektur

### Basisklasse

Alle Chunking-Implementierungen erben von der abstrakten Basisklasse `BaseChunker`:

```python
class BaseChunker(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Abstrakte Methode zur Aufteilung von Dokumenten in Chunks."""
        pass
```

### Token-Validierung

Die Basisklasse stellt Funktionen zur Token-Überwachung bereit:

- `count_tokens_estimate(text: str)`: Schätzt die Anzahl der Tokens unter Berücksichtigung von Subword-Tokenization
- `validate_chunk_tokens(chunk_content: str, chunk_id: str)`: Validiert Chunks gegen das 512-Token-Limit für Embedding-Modelle

Bei Überschreitung des Limits wird eine Warnung ausgegeben:
```
⚠️  WARNUNG: Chunk 'document_2' hat 547 Tokens (Limit: 512)
```

## Implementierte Chunking-Methoden

### 1. FixedSizeChunker

**Funktionsweise:** Teilt Dokumente in Chunks fester Zeichenlänge mit konfigurierbarem Overlap.

**Konfiguration:**
- `chunk_size` (default: 1280): Maximale Anzahl Zeichen pro Chunk
- `chunk_overlap` (default: 200): Anzahl überlappender Zeichen zwischen Chunks

**Anwendungsfall:** Baseline-Implementierung, vorhersagbare Chunk-Größen, einfache Konfiguration.

```python
chunker = FixedSizeChunker(chunk_size=1280, chunk_overlap=200)
chunks = chunker.split_documents(documents)
```

### 2. NLTKChunker

**Funktionsweise:** Satz-basierte Aufteilung unter Verwendung von NLTK's Sentence Tokenizer mit intelligentem Overlap-Mechanismus.

**Konfiguration:**
- `chunk_size` (default: 1280): Maximale Anzahl Zeichen pro Chunk
- `chunk_overlap` (default: 200): Overlap basierend auf vorherigen Sätzen

**Anwendungsfall:** Bewahrung der Satzintegrität, natürliche Textgrenzen, verbesserte semantische Kohärenz.

**Dependencies:** `nltk`

```python
chunker = NLTKChunker(chunk_size=1280, chunk_overlap=200)
chunks = chunker.split_documents(documents)
```

### 3. RecursiveChunker

**Funktionsweise:** Hierarchische Aufteilung anhand konfigurierbarer Separatoren (Absätze → Zeilen → Sätze → Wörter → Zeichen).

**Konfiguration:**
- `chunk_size` (default: 1280): Maximale Anzahl Zeichen pro Chunk
- `chunk_overlap` (default: 200): Anzahl überlappender Zeichen
- `separators` (default: `["\n\n", "\n", ". ", " ", ""]`): Hierarchie der Trennzeichen

**Anwendungsfall:** Intelligente Aufteilung entlang natürlicher Dokumentstrukturen, optimale Balance zwischen Chunk-Größe und semantischer Integrität.

```python
chunker = RecursiveChunker(
    chunk_size=1280, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = chunker.split_documents(documents)
```

### 4. SemanticChunker

**Funktionsweise:** Semantische Gruppierung von Sätzen durch Embedding-basiertes Clustering (Agglomerative Clustering mit Cosine-Distanz): Sätze werden in numerische Vektoren (Embeddings) umgewandelt, die ihre semantische Bedeutung repräsentieren. Anschließend werden ähnliche Sätze mittels Clustering-Algorithmus gruppiert, wobei die Ähnlichkeit durch Cosine-Distanz zwischen den Vektoren gemessen wird.

**Konfiguration:**
- `model_name` (default: `'sentence-transformers/all-MiniLM-L6-v2'`): Sentence Transformer Modell
- `chunk_size` (default: 3): Minimale Anzahl Sätze pro Chunk
- `distance_threshold` (default: 0.8): Cosine-Distanz-Schwellenwert für Clustering

**Anwendungsfall:** Thematisch kohärente Chunks, semantisch ähnliche Inhalte, experimentelle Optimierung für spezifische Domänen.

**Dependencies:** `sentence-transformers`, `scikit-learn`, `nltk`

```python
chunker = SemanticChunker(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    chunk_size=3,
    distance_threshold=0.8
)
chunks = chunker.split_documents(documents)
```

## Optimierung für Small Language Models (SLMs)

Alle Chunker sind für die Verwendung mit Small Language Models optimiert:

### Token-Limits
- **Embedding-Limit:** 512 Tokens pro Chunk (automatische Validierung)
- **Standard-Chunk-Größen:** 1280 Zeichen (~320-400 Tokens)
- **Konservative Schätzung:** 3-4 Zeichen pro Token bei deutschsprachigen Texten

### SLM-spezifische Anpassungen
- Kleinere Default-Chunk-Größen im Vergleich zu LLM-optimierten Systemen
- Engere semantische Gruppierung beim SemanticChunker
- Weniger Sätze pro Cluster für bessere Kontextkohärenz

## Metadaten

Jeder generierte Chunk enthält erweiterte Metadaten:

```python
{
    "chunk": 0,                    # Index des Chunks
    "chunk_count": 5,              # Gesamtanzahl Chunks des Dokuments
    "estimated_tokens": 287,       # Geschätzte Token-Anzahl
    "source": "original_metadata"  # Ursprüngliche Dokumentmetadaten
}
```

## Testing und Validierung

### Automatisierte Tests
Jeder Chunker enthält integrierte Testfunktionalität:

```python
if __name__ == "__main__":
    try:
        test_doc = Document(content="Extended test content...", metadata={}, id="test")
        chunker = ChunkerClass(optimized_parameters)
        chunks = chunker.split_documents([test_doc])
        print("ChunkerName abgeschlossen")
    except Exception as e:
        print(f"Fehler: {e}")
```

### Token-Validierungstests
Die Tests verwenden erweiterte Testdokumente zur Validierung der Token-Limits:
- Längere Texte zur Auslösung von Token-Warnungen
- Realistische Dokumentgrößen für DSGVO-Inhalte
- Validierung der Chunk-Qualität und -Konsistenz

### Verwendung der Tests
```bash
# Einzelner Chunker-Test
python src/rag/components/chunking/fixedsize_chunker.py

# Umfassender Chunker-Vergleich
python src/rag/components/chunking/test_chunker.py
```

## Installation der Dependencies

```bash
# Grundlegende NLP-Bibliotheken
pip install nltk

# Machine Learning für SemanticChunker
pip install scikit-learn sentence-transformers

# Alle Dependencies
pip install -r requirements.txt
```

## Best Practices

### Chunk-Größenwahl
- **DSGVO-Artikel:** 800-1280 Zeichen für vollständige Artikel-Abschnitte
- **Kurze Texte:** 512-800 Zeichen für bessere Granularität
- **Overlap:** 15-25% der Chunk-Größe für Kontext-Erhaltung

### Methodenwahl
- **FixedSizeChunker:** Baseline und Performance-Vergleiche
- **NLTKChunker:** Strukturierte Texte mit klaren Satzgrenzen
- **RecursiveChunker:** Gemischte Dokumenttypen mit variabler Struktur
- **SemanticChunker:** Thematisch komplexe Inhalte, experimentelle Optimierung

### Evaluations-Kriterien
- Token-Anzahl und -Verteilung
- Semantische Kohärenz der Chunks
- Verarbeitungsgeschwindigkeit
- Retrieval-Qualität in der nachgelagerten Pipeline

## Integration in die RAG-Pipeline

Das Chunking-Modul als erster Verarbeitungsschritt in RAG-Pipeline:

1. **Input:** Document-Objekte mit DSGVO-Inhalten
2. **Processing:** Chunking-Methode nach Konfiguration
3. **Output:** Validierte Chunks mit Token-Metadaten
4. **Weiterleitung:** An Embedding-Komponente für Vektorisierung

Die modulare Architektur ermöglicht den Vergleich verschiedener Strategien und experimentelle Optimierungen für bestmögliche RAG-Performance.