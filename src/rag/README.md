# ResearchRAG - Production-Ready RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system designed for scientific evaluation and experimentation with different language models, embedding strategies, and retrieval methods.

## 🚀 Features

- **Modular Architecture**: Easy exchange of LLMs, embeddings, chunking, and retrieval components
- **Production-Ready**: Robust error handling, caching, and comprehensive evaluation metrics
- **Multi-Model Support**: OpenAI API and local models (Ollama)
- **Smart Caching**: Automatic caching of chunks, embeddings, and evaluation results
- **Comprehensive Evaluation**: Precision/Recall, RAGAS-style metrics, timing, and performance analysis
- **Experiment Comparison**: Built-in tools to compare multiple RAG configurations
- **YAML Configuration**: Easy parameter tuning without code changes

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🏗️ Architecture

```
src/rag/
├── interfaces.py      # Abstract base classes
├── config.py         # Configuration management
├── factory.py        # Component factory
├── pipeline.py       # Main RAG pipeline
├── llms.py          # Language model implementations
├── embeddings.py    # Embedding model implementations
├── chunking.py      # Text chunking strategies
├── retrieval.py     # Retrieval methods
├── evaluation.py    # Evaluation metrics
├── evaluator.py     # Main evaluation orchestrator
├── compare_experiments.py  # Experiment comparison tool
├── cache.py         # Caching system
├── dataset.py       # Dataset handling
└── main.py          # CLI entry point
```

## ⚙️ Configuration

### Basic Configuration Structure

```yaml
# configs/baseline.yaml
llm:
  type: openai
  model: gpt-4o-mini
  temperature: 0.1
  max_tokens: 1000

embedding:
  type: openai
  model: text-embedding-ada-002

chunking:
  type: fixed-size
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  type: vector-similarity
  top_k: 5
  similarity_threshold: 0.0

evaluation:
  enabled_metrics:
    - precision-recall
    - timing
    - ragas
```

### Local Model Configuration

```yaml
# configs/mixtral_7b.yaml
llm:
  type: local
  model_name: mixtral:8x7b
  endpoint: http://localhost:11434
  temperature: 0.1
  max_tokens: 1000

embedding:
  type: sentence-transformers
  model_name: all-MiniLM-L6-v2
  device: cpu

# ... rest of configuration
```

## 🚀 Usage

### Single Experiment Evaluation

```bash
# Run baseline evaluation
python src/rag/main.py \
  --config configs/baseline.yaml \
  --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl \
  --num-qa 50

# Run local model evaluation
python src/rag/main.py \
  --config configs/mixtral_7b.yaml \
  --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl \
  --num-qa 50
```

### Compare Multiple Experiments

```bash
# Compare baseline and Mixtral configurations
python src/rag/compare_experiments.py \
  --configs configs/baseline.yaml configs/mixtral_7b.yaml \
  --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl \
  --num-qa 50

# Compare all configurations
python src/rag/compare_experiments.py \
  --configs configs/*.yaml \
  --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl \
  --num-qa 50 \
  --output-dir my_comparison_results
```

## 📊 Evaluation Metrics

### Core Metrics
- **Precision/Recall/F1**: Traditional information retrieval metrics
- **Sequence Similarity**: Text similarity using difflib
- **Semantic Similarity**: Word overlap with stemming approximation

### RAGAS-Style Metrics
- **Faithfulness**: How much the answer relies on provided context
- **Answer Relevance**: Relevance of answer to the question
- **Context Relevance**: Relevance of retrieved chunks to the question
- **Context Utilization**: How well the context is used in the answer

### Performance Metrics
- **Query Time**: Response generation time
- **Tokens per Second**: Processing speed
- **Response Quality**: Length, word count, average word length

## 💾 Caching System

The system automatically caches:
- **Chunks**: Text chunks with configuration-based hashing
- **Embeddings**: Vector representations of chunks
- **Evaluation Results**: Complete evaluation outputs

Cache files are stored in `cache/<experiment_name>/` and automatically reused when configurations match.

## 🔧 Production Features

### Error Handling
- Graceful fallbacks for API failures
- Comprehensive logging and error reporting
- Connection testing for local services

### Performance
- Efficient numpy-based vector operations
- Configurable chunking and retrieval parameters
- Automatic similarity threshold filtering

### Monitoring
- Detailed logging at all levels
- Performance metrics collection
- Experiment comparison reports

## 📁 Output Structure

```
project_root/
├── cache/                    # Cached components
│   ├── baseline/            # Baseline experiment cache
│   └── mixtral_7b/         # Mixtral experiment cache
├── comparison_results/       # Experiment comparison outputs
│   ├── comparison_results_YYYYMMDD_HHMMSS.json
│   ├── comparison_data_YYYYMMDD_HHMMSS.csv
│   └── comparison_report_YYYYMMDD_HHMMSS.txt
├── baseline_evaluation_results_YYYYMMDD_HHMMSS.json
├── baseline_evaluation_summary_YYYYMMDD_HHMMSS.json
├── mixtral_7b_evaluation_results_YYYYMMDD_HHMMSS.json
└── mixtral_7b_evaluation_summary_YYYYMMDD_HHMMSS.json
```

## 🧪 Experiment Examples

### Baseline (GPT-4o-mini)
- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: OpenAI text-embedding-ada-002
- **Chunking**: Fixed-size (1000 chars, 200 overlap)
- **Retrieval**: Vector similarity with cosine distance

### Mixtral-8x7B (Local)
- **LLM**: Local Mixtral-8x7B via Ollama
- **Embedding**: Sentence Transformers (all-MiniLM-L6-v2)
- **Chunking**: Fixed-size (1000 chars, 200 overlap)
- **Retrieval**: Vector similarity with cosine distance

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check your `.env` file and API key
2. **Local Model Connection**: Ensure Ollama is running on localhost:11434
3. **Memory Issues**: Reduce chunk size or number of QA pairs
4. **Cache Issues**: Use `--clear-cache` flag or manually delete cache directory

### Performance Tuning

- **Chunk Size**: Smaller chunks = better precision, larger chunks = better context
- **Top-K**: Higher values = more context, lower values = faster retrieval
- **Similarity Threshold**: Higher values = more relevant results, lower values = more results

## 📈 Next Steps

1. **Run Baseline**: Test with GPT-4o-mini configuration
2. **Test Local Models**: Ensure Ollama is running and accessible
3. **Compare Results**: Use comparison tool to analyze performance
4. **Tune Parameters**: Adjust chunking, retrieval, and evaluation parameters
5. **Scale Up**: Increase QA pairs for more comprehensive evaluation

## 🤝 Contributing

The system is designed for easy extension:
- Add new LLM providers in `llms.py`
- Implement new embedding models in `embeddings.py`
- Create custom chunking strategies in `chunking.py`
- Develop novel retrieval methods in `retrieval.py`
- Add evaluation metrics in `evaluation.py`

## 📄 License

This project is part of the ResearchRAG system for scientific evaluation of RAG configurations.
