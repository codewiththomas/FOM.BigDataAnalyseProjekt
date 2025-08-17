### TODOs 16.08.2025

- Crawler
  [ ] DSGVO zu Artikeln zusammenfassen [TB]
  [ ] Unterabsätze sind doppelt
- Chunker
  [ ] RecursiveChunker implementieren
- FAISS als Datenbank hinterlegen
- Kapitel Design schreiben
  [ ] Crawler
  [ ] Architektur
- Evaluation
  [ ] Kapitel 5 Evaluationsdesign beschreiben (welche Metriken, was haben vor)
  [ ] testen mit Baseline
- Notizen bzgl. Hyperparameter in OneNote

Beispiel-Modelle:
mistral-nemo:12b                         e7e06d107c6c    7.1 GB    28 minutes ago
lukasmalkmus/llama3-sauerkraut:latest    755e202f8556    6.6 GB    3 hours ago
llama3.2:1b                              baf6a787fdff    1.3 GB    3 hours ago
llama3.2:3b                              a80c4f17acd5    2.0 GB    3 hours ago
gpt-oss:20b                              aa4295ac10c3    13 GB     3 hours ago
qwen3:8b                                 500a1f067a9f    5.2 GB    23 hours ago
mixtral:8x7b                             a3b6bef0f836    26 GB     43 hours ago
qwen2.5:7b-instruct                      845dbda0ea48    4.7 GB    3 weeks ago
gemma3:1b                                8648f39daa8f    815 MB    3 weeks ago
gemma3:latest                            a2af6cc3eb7f    3.3 GB    3 weeks ago

###

Crawler:
- Unterabsätze sind doppelt
- Anzahl Zeichen mit angeben

Weitere Modelle:
- Qwen2.5
- Qwen 3
- GPT-OSS-20B
- Llama-3
- Llama-3-Sauerkraut
- GPT-5-nano

Chunker
- FixedSizeChunking (mit/ohne Overlap)
- SemanticChunking
- RecursiveChunker
(- NLTK => Satzbasierte Aufteilung )
(- SentenceChunking (was passiert eigentlich, wenn Sätze länger sind als der Chunk?))

Ebeddings
- text-embedding-ada-002
- all-MiniLM-L6-v2

Datenbank
- FAISS mit HNSW zur Indexierung => FIX
- (Chroma)
- (QDRANT)
- (HNSWLIB)

Suche
- BM25
- Cosine Similarity

DSGVO
- Dat

Retrieval nutzt MD5 statt die Embeddings-Klasse






