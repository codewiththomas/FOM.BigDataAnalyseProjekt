# ResearchRAG System Architecture

## Overview

ResearchRAG is a modular Retrieval-Augmented Generation (RAG) system designed for scientific research and experimentation. The system allows researchers to easily swap different components (chunkers, embeddings, vector stores, and language models) to study their impact on RAG performance.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ResearchRAG System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Data Layer    │  │  Core Pipeline  │  │  Evaluation     │ │
│  │                 │  │                 │  │  Framework      │ │
│  │ • Raw Data      │  │ • RAG Pipeline  │  │ • Evaluators    │ │
│  │ • Processed     │  │ • Component     │  │ • Metrics       │ │
│  │ • QA Pairs      │  │   Loader        │  │ • Experiments   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Component Layer                              │ │
│  │                                                             │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │ │
│  │ │  Chunkers   │ │ Embeddings  │ │Vector Stores│ │  LLMs   │ │ │
│  │ │             │ │             │ │             │ │         │ │ │
│  │ │ • Line      │ │ • OpenAI    │ │ • InMemory  │ │ • OpenAI│ │ │
│  │ │ • Recursive │ │ • Sentence  │ │ • Chroma    │ │ • Ollama│ │ │
│  │ │ • Semantic  │ │   Transformer│ │ • FAISS     │ │         │ │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Base Classes                              │ │
│  │                                                             │ │
│  │ BaseChunker | BaseEmbedding | BaseVectorStore | BaseLLM    │ │
│  │                      BaseEvaluator                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Component Layer

**Base Classes:**
- `BaseChunker`: Abstract interface for text chunking strategies
- `BaseEmbedding`: Abstract interface for text embedding models
- `BaseVectorStore`: Abstract interface for vector storage and retrieval
- `BaseLanguageModel`: Abstract interface for language models
- `BaseEvaluator`: Abstract interface for evaluation metrics

**Implementations:**

**Chunkers:**
- `LineChunker`: Simple line-based chunking
- `RecursiveChunker`: Hierarchical chunking with multiple separators
- `SemanticChunker`: Semantic similarity-based chunking

**Embeddings:**
- `OpenAIEmbedding`: OpenAI embedding models (text-embedding-3-small/large)
- `SentenceTransformerEmbedding`: Local sentence transformer models

**Vector Stores:**
- `InMemoryVectorStore`: Simple in-memory vector storage
- `ChromaVectorStore`: Persistent storage with ChromaDB
- `FAISSVectorStore`: High-performance vector search with FAISS

**Language Models:**
- `OpenAILanguageModel`: OpenAI GPT models
- `OllamaLanguageModel`: Local models via Ollama

#### 2. Core Pipeline

**RAGPipeline:**
- Orchestrates the entire RAG process
- Loads components based on configuration
- Handles document indexing and query processing
- Provides unified interface for different component combinations

**ComponentLoader:**
- Dynamic component loading and registration
- Configuration-based component instantiation
- Supports plugin architecture for extensibility

#### 3. Evaluation Framework

**Evaluators:**
- `RetrievalEvaluator`: Precision@k, Recall@k, F1@k, MRR, NDCG
- `GenerationEvaluator`: ROUGE-L, BLEU, Exact Match, Semantic Similarity
- `PerformanceEvaluator`: Latency, Throughput, Memory Usage, Cost
- `RAGEvaluator`: End-to-end RAG evaluation with combined metrics

**ExperimentRunner:**
- Systematic experiment execution
- Configuration comparison
- Ablation studies
- Statistical analysis and reporting

## Data Flow

### 1. Document Indexing Flow

```
Raw Documents → Chunker → Embedding Model → Vector Store
     │              │            │              │
     │              │            │              ├─ Store embeddings
     │              │            │              ├─ Store metadata
     │              │            │              └─ Build index
     │              │            └─ Generate embeddings
     │              └─ Split into chunks
     └─ Load text data
```

### 2. Query Processing Flow

```
User Query → Embedding Model → Vector Store → Retrieved Chunks
    │              │                │              │
    │              │                │              └─ Top-k similar chunks
    │              │                └─ Similarity search
    │              └─ Generate query embedding
    └─ Process query

Retrieved Chunks + Query → Language Model → Generated Answer
         │                      │              │
         │                      │              └─ Final response
         │                      └─ Generate response
         └─ Create context prompt
```

### 3. Evaluation Flow

```
QA Dataset → RAG Pipeline → Predictions → Evaluators → Metrics
    │              │             │            │           │
    │              │             │            │           ├─ Retrieval metrics
    │              │             │            │           ├─ Generation metrics
    │              │             │            │           ├─ Performance metrics
    │              │             │            │           └─ Combined RAG score
    │              │             │            └─ Apply evaluation
    │              │             └─ Generated answers
    │              └─ Process questions
    └─ Load questions and gold answers
```

## Configuration System

### Configuration Structure

```python
{
    "chunker": {
        "type": "recursive_chunker",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", ". ", " "]
    },
    "embedding": {
        "type": "sentence_transformer",
        "model": "all-MiniLM-L6-v2",
        "normalize_embeddings": true
    },
    "vector_store": {
        "type": "chroma",
        "collection_name": "rag_documents",
        "persist_directory": "./data/chroma_db"
    },
    "language_model": {
        "type": "ollama",
        "model": "llama3.2",
        "temperature": 0.1,
        "max_tokens": 500
    }
}
```

### Component Registration

Components are registered with the ComponentLoader:

```python
# Register component
loader.register_component("chunker", "recursive", RecursiveChunker)

# Load component from config
chunker = loader.load_component("chunker", config["chunker"])
```

## Extensibility

### Adding New Components

1. **Create Component Class:**
   ```python
   class NewChunker(BaseChunker):
       def chunk_text(self, text: str) -> List[str]:
           # Implementation
           pass
   ```

2. **Register Component:**
   ```python
   loader.register_component("chunker", "new_chunker", NewChunker)
   ```

3. **Use in Configuration:**
   ```python
   config = {
       "chunker": {
           "type": "new_chunker",
           "param1": "value1"
       }
   }
   ```

### Plugin Architecture

The system supports plugin-based extensions:

```python
# Plugin structure
plugins/
├── __init__.py
├── custom_chunker.py
├── custom_embedding.py
└── custom_evaluator.py
```

## Performance Considerations

### Memory Management

- **Lazy Loading**: Components are loaded only when needed
- **Batch Processing**: Large datasets are processed in batches
- **Memory Monitoring**: Built-in memory usage tracking
- **Cleanup**: Automatic cleanup of unused resources

### Scalability

- **Parallel Processing**: Multi-threaded embedding generation
- **GPU Support**: CUDA acceleration for compatible models
- **Distributed Storage**: Support for distributed vector stores
- **Caching**: Intelligent caching of embeddings and results

### Optimization Strategies

1. **Embedding Caching**: Store computed embeddings to avoid recomputation
2. **Index Optimization**: Use appropriate vector store configurations
3. **Batch Operations**: Process multiple queries simultaneously
4. **Model Quantization**: Use quantized models for faster inference

## Security Considerations

### Data Privacy

- **Local Processing**: Support for completely local pipelines
- **API Key Management**: Secure handling of API credentials
- **Data Encryption**: Optional encryption for stored embeddings
- **Access Control**: Role-based access to different components

### Model Security

- **Model Validation**: Verification of model integrity
- **Sandboxing**: Isolated execution environments
- **Audit Logging**: Comprehensive logging of all operations
- **Rate Limiting**: Protection against abuse

## Monitoring and Logging

### Metrics Collection

```python
# Performance metrics
{
    "latency": 0.45,
    "throughput": 12.3,
    "memory_usage": 512.5,
    "cost_per_query": 0.001
}

# Quality metrics
{
    "precision_at_5": 0.85,
    "recall_at_5": 0.78,
    "rouge_l": 0.72,
    "bleu_score": 0.68
}
```

### Logging Framework

- **Structured Logging**: JSON-formatted logs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Component Tracing**: Track execution through components
- **Performance Profiling**: Detailed timing information

## Deployment Options

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with local models
python -m src.ResearchRAG --config local_config.json
```

### Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install package
!pip install -e .

# Run experiments
%run src/ResearchRAG.ipynb
```

### Production Deployment

```bash
# Docker deployment
docker build -t research-rag .
docker run -p 8000:8000 research-rag

# Kubernetes deployment
kubectl apply -f k8s/deployment.yaml
```

## Testing Strategy

### Unit Tests

```python
# Test component interfaces
def test_chunker_interface():
    chunker = LineChunker()
    result = chunker.chunk_text("Test text")
    assert isinstance(result, list)
```

### Integration Tests

```python
# Test full pipeline
def test_rag_pipeline():
    pipeline = RAGPipeline(config)
    pipeline.index_documents(documents)
    result = pipeline.query("Test question")
    assert result is not None
```

### Performance Tests

```python
# Benchmark components
def test_performance():
    evaluator = PerformanceEvaluator()
    metrics = evaluator.benchmark_component(component)
    assert metrics["latency"] < 1.0
```

## Future Enhancements

### Planned Features

1. **Multi-modal Support**: Support for images and other media types
2. **Federated Learning**: Distributed training capabilities
3. **Real-time Updates**: Live document indexing and updates
4. **Advanced Metrics**: More sophisticated evaluation metrics
5. **Web Interface**: Browser-based configuration and monitoring

### Research Directions

1. **Adaptive Chunking**: Dynamic chunk sizing based on content
2. **Hybrid Retrieval**: Combining multiple retrieval strategies
3. **Reinforcement Learning**: Self-improving RAG systems
4. **Explainable AI**: Interpretable retrieval and generation
5. **Multi-language Support**: Cross-lingual RAG capabilities

## Conclusion

The ResearchRAG system provides a flexible, extensible framework for RAG research and experimentation. Its modular architecture allows researchers to easily compare different approaches and identify optimal configurations for their specific use cases. The comprehensive evaluation framework ensures rigorous assessment of system performance across multiple dimensions.

For more detailed information, see:
- [API Reference](api_reference.md)
- [Experiment Logs](experiment_logs.md)
- [Results Analysis](results_analysis.md)