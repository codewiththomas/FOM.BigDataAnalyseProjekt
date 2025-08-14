# ResearchRAG - Modular RAG Evaluation System

A modular, configurable RAG (Retrieval-Augmented Generation) system designed for scientific evaluation and experimentation with different components.

## 🎯 Features

- **Modular Architecture**: Easily swap LLMs, embeddings, chunking strategies, and retrieval methods
- **YAML Configuration**: Simple configuration files for different experiments
- **Automatic Evaluation**: Built-in evaluation metrics (Precision, Recall, F1, RAGAS-style, timing)
- **DSGVO Dataset**: Pre-configured for DSGVO legal text evaluation
- **Multiple Models**: Support for OpenAI API and local models (GPT-OSS-20B, Mixtral-7B, Qwen3, Llama3-Sauerkraut)

## 🏗️ Architecture

```
ResearchRAG/
├── interfaces.py      # Abstract interfaces for all components
├── pipeline.py        # Main RAG pipeline orchestrator
├── config.py          # YAML configuration manager
├── factory.py         # Component factory
├── llms.py           # LLM implementations (OpenAI, Local)
├── embeddings.py     # Embedding implementations (OpenAI, Sentence-Transformers)
├── chunking.py       # Chunking strategies (Fixed-size, Semantic)
├── retrieval.py      # Retrieval methods (Vector similarity, Hybrid)
├── evaluation.py     # Evaluation metrics
├── dataset.py        # DSGVO dataset management
├── evaluator.py      # Main evaluation orchestrator
└── main.py           # CLI interface
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_rag.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run Baseline Evaluation

```bash
python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl
```

## 📋 Configuration

### Baseline Configuration (GPT-4o-mini)

```yaml
llm:
  type: openai
  model: gpt-4o-mini
  temperature: 0.1

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
```

### Local Models Configuration

```yaml
llm:
  type: local
  model_name: gpt-oss-20b
  endpoint: http://localhost:8000
  api_type: ollama

embedding:
  type: sentence-transformers
  model_name: all-MiniLM-L6-v2
  device: cpu
```

## 🔬 Experiments

### 1. Different Language Models

- **GPT-4o-mini**: Baseline reference model
- **GPT-OSS-20B**: Open-source 20B parameter model
- **Mixtral-7B**: Mixture of experts model
- **Qwen3**: Alibaba's Qwen model
- **Llama3-Sauerkraut**: German-optimized Llama model

### 2. Embedding & Chunking Experiments

- **Fixed-size chunking**: Different chunk sizes (500, 1000, 2000)
- **Semantic chunking**: Natural boundary-based chunking
- **Sentence-based**: Sentence-level chunking
- **Embedding models**: OpenAI vs. Sentence-Transformers

### 3. Retrieval Techniques

- **Vector similarity**: Different top-k values (3, 5, 10)
- **Similarity thresholds**: Filtering by relevance
- **Hybrid retrieval**: Vector + keyword combinations

## 📊 Evaluation Metrics

### Traditional Metrics
- **Precision**: How many retrieved chunks are relevant
- **Recall**: How many relevant chunks were retrieved
- **F1**: Harmonic mean of precision and recall

### RAG-Specific Metrics
- **Faithfulness**: How much response is based on retrieved chunks
- **Answer Relevance**: How relevant response is to query
- **Context Relevance**: How relevant retrieved chunks are to query

### Performance Metrics
- **Query Time**: Total time per query
- **Tokens per Second**: Generation speed
- **Response Length**: Output size

## 🎮 Usage Examples

### Run Different Configurations

```bash
# Baseline evaluation
python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl

# Local models evaluation
python -m rag.main --config configs/local_models.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl

# Chunking experiments
python -m rag.main --config configs/chunking_experiments.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl

# Retrieval experiments
python -m rag.main --config configs/retrieval_experiments.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl
```

### Custom Evaluation

```bash
# Evaluate with 100 QA pairs
python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --num-qa 100

# Run without saving results
python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --no-save

# Verbose logging
python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --verbose
```

## 🔧 Customization

### Adding New LLM Types

1. Implement the `LLMInterface` in `llms.py`
2. Add to `LLMFactory.create_llm()`
3. Create configuration file

### Adding New Embedding Models

1. Implement the `EmbeddingInterface` in `embeddings.py`
2. Add to `EmbeddingFactory.create_embedding()`
3. Update configuration

### Adding New Evaluation Metrics

1. Implement the `EvaluationInterface` in `evaluation.py`
2. Add to `EvaluationManager._setup_evaluators()`
3. Enable in configuration

## 📁 Output Files

The system generates:
- `evaluation_results_YYYYMMDD_HHMMSS.json`: Detailed results for each QA pair
- `evaluation_summary_YYYYMMDD_HHMMSS.json`: Summary statistics and metrics

## 🚧 Current Limitations

- **Local LLMs**: Placeholder implementations - need integration with Ollama, vLLM, etc.
- **Vector Search**: Simplified similarity calculation - could use FAISS, Chroma, etc.
- **Advanced Metrics**: RAGAS implementation is simplified - could use full RAGAS library

## 🔮 Future Enhancements

- **Real Vector Database**: Integration with FAISS, Chroma, Pinecone
- **Advanced Chunking**: NLP-based semantic chunking
- **Full RAGAS**: Complete RAGAS metric implementation
- **Batch Processing**: Parallel evaluation for faster results
- **Web Interface**: Web-based configuration and monitoring

## 🤝 Contributing

1. Follow the modular interface design
2. Add comprehensive logging
3. Include error handling
4. Update configuration examples
5. Add tests for new components

## 📄 License

This project is part of the FOM Big Data Analysis Project.
