# ResearchRAG API Reference

## Overview

This document provides comprehensive API documentation for all components of the ResearchRAG system. The API is organized into the following main categories:

- [Core Components](#core-components)
- [Chunkers](#chunkers)
- [Embeddings](#embeddings)
- [Vector Stores](#vector-stores)
- [Language Models](#language-models)
- [Evaluators](#evaluators)
- [Experiment Framework](#experiment-framework)

## Core Components

### RAGPipeline

The main orchestrator for the RAG process.

```python
class RAGPipeline:
    def __init__(self, config: Dict[str, Any])
    def index_documents(self, documents: List[str]) -> bool
    def query(self, question: str) -> str
    def get_pipeline_info(self) -> Dict[str, Any]
    def clear_index(self) -> bool
```

#### Methods

**`__init__(config: Dict[str, Any])`**
- Initializes the RAG pipeline with the given configuration
- **Parameters:**
  - `config`: Configuration dictionary containing component settings
- **Raises:**
  - `ValueError`: If configuration is invalid
  - `RuntimeError`: If components cannot be loaded

**`index_documents(documents: List[str]) -> bool`**
- Indexes a list of documents into the vector store
- **Parameters:**
  - `documents`: List of text documents to index
- **Returns:**
  - `bool`: True if indexing was successful
- **Raises:**
  - `RuntimeError`: If pipeline is not properly initialized

**`query(question: str) -> str`**
- Processes a query and returns a generated answer
- **Parameters:**
  - `question`: The question to answer
- **Returns:**
  - `str`: Generated answer based on retrieved context
- **Raises:**
  - `RuntimeError`: If no documents are indexed

**`get_pipeline_info() -> Dict[str, Any]`**
- Returns information about the current pipeline configuration
- **Returns:**
  - `Dict[str, Any]`: Pipeline configuration and status

**`clear_index() -> bool`**
- Clears all indexed documents from the vector store
- **Returns:**
  - `bool`: True if clearing was successful

### ComponentLoader

Manages dynamic loading and registration of components.

```python
class ComponentLoader:
    def register_component(self, component_type: str, name: str, component_class: Type)
    def load_component(self, component_type: str, config: Dict[str, Any]) -> Any
    def get_registered_components(self) -> Dict[str, Dict[str, Type]]
    def register_evaluator(self, name: str, evaluator_class: Type)
    def load_evaluator(self, name: str, config: Dict[str, Any]) -> BaseEvaluator
```

#### Methods

**`register_component(component_type: str, name: str, component_class: Type)`**
- Registers a new component implementation
- **Parameters:**
  - `component_type`: Type of component ('chunker', 'embedding', 'vector_store', 'language_model')
  - `name`: Unique name for the component
  - `component_class`: Class implementing the component

**`load_component(component_type: str, config: Dict[str, Any]) -> Any`**
- Loads a component instance based on configuration
- **Parameters:**
  - `component_type`: Type of component to load
  - `config`: Configuration dictionary for the component
- **Returns:**
  - Component instance
- **Raises:**
  - `ValueError`: If component type or name is not registered

## Chunkers

### BaseChunker

Abstract base class for all chunkers.

```python
class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200)
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]
    def get_chunk_metadata(self, chunk: str, chunk_index: int) -> Dict[str, Any]
    def estimate_chunks(self, text: str) -> int
    def validate_chunk_size(self, chunk_size: int) -> bool
```

### LineChunker

Simple line-based text chunker.

```python
class LineChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 preserve_line_breaks: bool = True)
    def chunk_text(self, text: str) -> List[str]
```

#### Parameters
- `chunk_size`: Maximum size of each chunk in characters
- `chunk_overlap`: Number of characters to overlap between chunks
- `preserve_line_breaks`: Whether to preserve line breaks in chunks

### RecursiveChunker

Hierarchical chunker using multiple separators.

```python
class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None, keep_separator: bool = True)
    def chunk_text(self, text: str) -> List[str]
```

#### Parameters
- `chunk_size`: Maximum size of each chunk in characters
- `chunk_overlap`: Number of characters to overlap between chunks
- `separators`: List of separators to try in order of preference
- `keep_separator`: Whether to keep separators in the chunks

### SemanticChunker

Semantic similarity-based chunker.

```python
class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 similarity_threshold: float = 0.7, min_sentences_per_chunk: int = 2,
                 embedding_model: Optional[str] = None)
    def chunk_text(self, text: str) -> List[str]
```

#### Parameters
- `chunk_size`: Maximum size of each chunk in characters
- `chunk_overlap`: Number of characters to overlap between chunks
- `similarity_threshold`: Threshold for semantic similarity (0-1)
- `min_sentences_per_chunk`: Minimum number of sentences per chunk
- `embedding_model`: Name of the embedding model to use

## Embeddings

### BaseEmbedding

Abstract base class for all embedding models.

```python
class BaseEmbedding(ABC):
    def __init__(self)
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray
    def embed_query(self, query: str) -> np.ndarray
    def get_embedding_dimension(self) -> int
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float
```

### OpenAIEmbedding

OpenAI embedding model implementation.

```python
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None,
                 dimensions: Optional[int] = None, batch_size: int = 100)
    def embed_texts(self, texts: List[str]) -> np.ndarray
    def embed_query(self, query: str) -> np.ndarray
    def get_embedding_dimension(self) -> int
    def calculate_cost(self, num_tokens: int) -> float
```

#### Parameters
- `model`: OpenAI embedding model name
- `api_key`: OpenAI API key (optional if set in environment)
- `dimensions`: Embedding dimensions (model-specific)
- `batch_size`: Batch size for processing multiple texts

### SentenceTransformerEmbedding

Local sentence transformer embedding implementation.

```python
class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None,
                 normalize_embeddings: bool = True, batch_size: int = 32)
    def embed_texts(self, texts: List[str]) -> np.ndarray
    def embed_query(self, query: str) -> np.ndarray
    def get_embedding_dimension(self) -> int
    def get_available_models(self) -> List[str]
    def benchmark_model(self, test_texts: List[str]) -> Dict[str, Any]
```

#### Parameters
- `model_name`: Name of the sentence transformer model
- `device`: Device to run the model on ('cuda', 'cpu', or None for auto)
- `normalize_embeddings`: Whether to normalize embeddings to unit length
- `batch_size`: Batch size for processing multiple texts

## Vector Stores

### BaseVectorStore

Abstract base class for all vector stores.

```python
class BaseVectorStore(ABC):
    def __init__(self)
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None) -> List[str]
    @abstractmethod
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]
    def get_all_documents(self) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]
    def delete_documents(self, ids: List[str]) -> bool
    def get_collection_stats(self) -> Dict[str, Any]
```

### InMemoryVectorStore

Simple in-memory vector store implementation.

```python
class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, similarity_metric: str = "cosine")
    def add_texts(self, texts: List[str], embeddings: np.ndarray,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None) -> List[str]
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]
    def clear_collection(self) -> bool
```

#### Parameters
- `similarity_metric`: Similarity metric to use ('cosine', 'euclidean', 'dot_product')

### ChromaVectorStore

ChromaDB-based persistent vector store.

```python
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "rag_documents",
                 persist_directory: Optional[str] = None,
                 embedding_function: Optional[Any] = None,
                 distance_metric: str = "cosine")
    def add_texts(self, texts: List[str], embeddings: np.ndarray,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None) -> List[str]
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]
    def backup_collection(self, backup_path: str) -> bool
    def restore_collection(self, backup_path: str) -> bool
```

#### Parameters
- `collection_name`: Name of the ChromaDB collection
- `persist_directory`: Directory to persist the database
- `embedding_function`: Custom embedding function for ChromaDB
- `distance_metric`: Distance metric ('cosine', 'l2', 'ip')

### FAISSVectorStore

FAISS-based high-performance vector store.

```python
class FAISSVectorStore(BaseVectorStore):
    def __init__(self, embedding_dimension: int, index_type: str = "IndexFlatIP",
                 metric_type: str = "cosine", nlist: int = 100, use_gpu: bool = False)
    def add_texts(self, texts: List[str], embeddings: np.ndarray,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None) -> List[str]
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]
    def save_index(self, filepath: str) -> bool
    def load_index(self, filepath: str) -> bool
    def train_index(self, training_embeddings: np.ndarray) -> bool
```

#### Parameters
- `embedding_dimension`: Dimension of the embeddings
- `index_type`: Type of FAISS index ('IndexFlatIP', 'IndexIVFFlat', 'IndexHNSWFlat')
- `metric_type`: Distance metric ('cosine', 'l2', 'ip')
- `nlist`: Number of clusters for IVF indexes
- `use_gpu`: Whether to use GPU acceleration

## Language Models

### BaseLanguageModel

Abstract base class for all language models.

```python
class BaseLanguageModel(ABC):
    def __init__(self)
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str
    def generate_with_context(self, context: str, question: str, **kwargs) -> str
    def get_model_info(self) -> Dict[str, Any]
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float
```

### OpenAILanguageModel

OpenAI language model implementation.

```python
class OpenAILanguageModel(BaseLanguageModel):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 500)
    def generate(self, prompt: str, **kwargs) -> str
    def generate_with_context(self, context: str, question: str, **kwargs) -> str
    def get_model_info(self) -> Dict[str, Any]
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float
    def get_available_models(self) -> List[str]
```

#### Parameters
- `model`: OpenAI model name
- `api_key`: OpenAI API key (optional if set in environment)
- `temperature`: Sampling temperature (0.0 to 1.0)
- `max_tokens`: Maximum number of tokens to generate

### OllamaLanguageModel

Ollama local language model implementation.

```python
class OllamaLanguageModel(BaseLanguageModel):
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434",
                 temperature: float = 0.1, max_tokens: int = 500)
    def generate(self, prompt: str, **kwargs) -> str
    def generate_with_context(self, context: str, question: str, **kwargs) -> str
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]
    def get_available_models(self) -> List[str]
    def health_check(self) -> Dict[str, Any]
    def benchmark_model(self, test_prompts: List[str]) -> Dict[str, Any]
```

#### Parameters
- `model_name`: Name of the Ollama model to use
- `base_url`: Base URL for Ollama API
- `temperature`: Sampling temperature (0.0 to 1.0)
- `max_tokens`: Maximum number of tokens to generate

## Evaluators

### BaseEvaluator

Abstract base class for all evaluators.

```python
class BaseEvaluator(ABC):
    def __init__(self, name: str)
    @abstractmethod
    def evaluate(self, predictions: List[str], ground_truth: List[str],
                 contexts: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]
    def batch_evaluate(self, predictions: List[str], ground_truth: List[str],
                      batch_size: int = 32, **kwargs) -> Dict[str, Any]
    def save_results(self, results: Dict[str, Any], filepath: str)
    def load_results(self, filepath: str) -> Dict[str, Any]
```

### RetrievalEvaluator

Evaluates retrieval performance.

```python
class RetrievalEvaluator(BaseEvaluator):
    def __init__(self, k_values: List[int] = [1, 3, 5, 10])
    def evaluate(self, predictions: List[str], ground_truth: List[str],
                 contexts: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float
    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float
    def ndcg_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float
```

#### Parameters
- `k_values`: List of k values to evaluate for Precision@k and Recall@k

### GenerationEvaluator

Evaluates text generation quality.

```python
class GenerationEvaluator(BaseEvaluator):
    def __init__(self)
    def evaluate(self, predictions: List[str], ground_truth: List[str],
                 contexts: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]
    def rouge_l_score(self, prediction: str, reference: str) -> float
    def bleu_score(self, prediction: str, reference: str) -> float
    def exact_match(self, prediction: str, reference: str) -> bool
    def semantic_similarity(self, prediction: str, reference: str) -> float
```

### PerformanceEvaluator

Evaluates system performance metrics.

```python
class PerformanceEvaluator(BaseEvaluator):
    def __init__(self)
    def evaluate(self, predictions: List[str], ground_truth: List[str],
                 contexts: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]
    def measure_latency(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]
    def measure_throughput(self, func: Callable, inputs: List[Any], batch_size: int = 1) -> float
    def measure_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]
    def performance_decorator(self, func: Callable) -> Callable
```

### RAGEvaluator

Comprehensive end-to-end RAG evaluation.

```python
class RAGEvaluator(BaseEvaluator):
    def __init__(self, retrieval_evaluator: RetrievalEvaluator,
                 generation_evaluator: GenerationEvaluator,
                 performance_evaluator: PerformanceEvaluator)
    def evaluate(self, predictions: List[str], ground_truth: List[str],
                 contexts: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]
    def faithfulness_score(self, prediction: str, context: str) -> float
    def groundedness_score(self, prediction: str, context: str) -> float
    def answer_relevance_score(self, prediction: str, question: str) -> float
    def calculate_rag_score(self, retrieval_metrics: Dict, generation_metrics: Dict,
                           performance_metrics: Dict) -> float
```

## Experiment Framework

### ExperimentRunner

Manages and executes experiments.

```python
class ExperimentRunner:
    def __init__(self, evaluators: Dict[str, BaseEvaluator])
    def run_experiment(self, config: Dict[str, Any], qa_pairs: List[Dict[str, Any]],
                      experiment_name: str = "default") -> Dict[str, Any]
    def compare_configurations(self, configs: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]],
                              experiment_name: str = "comparison") -> Dict[str, Any]
    def run_ablation_study(self, base_config: Dict[str, Any], component_variations: Dict[str, List[Dict[str, Any]]],
                          qa_pairs: List[Dict[str, Any]], experiment_name: str = "ablation") -> Dict[str, Any]
    def benchmark_performance(self, configs: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]],
                             experiment_name: str = "benchmark") -> Dict[str, Any]
    def save_results(self, results: Dict[str, Any], filepath: str)
    def load_results(self, filepath: str) -> Dict[str, Any]
    def generate_report(self, results: Dict[str, Any], output_path: str)
```

#### Methods

**`run_experiment(config, qa_pairs, experiment_name="default")`**
- Runs a single experiment with the given configuration
- **Parameters:**
  - `config`: RAG pipeline configuration
  - `qa_pairs`: List of question-answer pairs for evaluation
  - `experiment_name`: Name for the experiment
- **Returns:**
  - `Dict[str, Any]`: Experiment results with metrics

**`compare_configurations(configs, qa_pairs, experiment_name="comparison")`**
- Compares multiple configurations on the same dataset
- **Parameters:**
  - `configs`: List of configurations to compare
  - `qa_pairs`: List of question-answer pairs for evaluation
  - `experiment_name`: Name for the comparison experiment
- **Returns:**
  - `Dict[str, Any]`: Comparison results with rankings

**`run_ablation_study(base_config, component_variations, qa_pairs, experiment_name="ablation")`**
- Runs ablation study by varying individual components
- **Parameters:**
  - `base_config`: Base configuration
  - `component_variations`: Dictionary of component variations
  - `qa_pairs`: List of question-answer pairs for evaluation
  - `experiment_name`: Name for the ablation study
- **Returns:**
  - `Dict[str, Any]`: Ablation study results

## Configuration Schema

### Pipeline Configuration

```python
{
    "chunker": {
        "type": str,  # Component type identifier
        "chunk_size": int,  # Maximum chunk size in characters
        "chunk_overlap": int,  # Overlap between chunks
        # Additional component-specific parameters
    },
    "embedding": {
        "type": str,  # Component type identifier
        "model": str,  # Model name or path
        "dimensions": int,  # Embedding dimensions (optional)
        # Additional component-specific parameters
    },
    "vector_store": {
        "type": str,  # Component type identifier
        "similarity_metric": str,  # Similarity metric
        "top_k": int,  # Number of documents to retrieve
        # Additional component-specific parameters
    },
    "language_model": {
        "type": str,  # Component type identifier
        "model": str,  # Model name
        "temperature": float,  # Sampling temperature
        "max_tokens": int,  # Maximum tokens to generate
        # Additional component-specific parameters
    }
}
```

### Evaluation Configuration

```python
{
    "evaluators": {
        "retrieval": {
            "k_values": List[int],  # K values for Precision@k, Recall@k
            "enable_statistical_tests": bool
        },
        "generation": {
            "metrics": List[str],  # Metrics to compute
            "reference_free": bool  # Whether to use reference-free metrics
        },
        "performance": {
            "measure_latency": bool,
            "measure_throughput": bool,
            "measure_memory": bool
        }
    },
    "experiment": {
        "name": str,  # Experiment name
        "description": str,  # Experiment description
        "save_results": bool,  # Whether to save results
        "output_directory": str  # Output directory for results
    }
}
```

## Error Handling

### Common Exceptions

```python
class RAGException(Exception):
    """Base exception for RAG-related errors"""
    pass

class ComponentNotFoundError(RAGException):
    """Raised when a component is not found"""
    pass

class ConfigurationError(RAGException):
    """Raised when configuration is invalid"""
    pass

class EvaluationError(RAGException):
    """Raised when evaluation fails"""
    pass

class ModelNotAvailableError(RAGException):
    """Raised when a model is not available"""
    pass
```

### Error Handling Best Practices

1. **Always handle exceptions** when loading components
2. **Validate configurations** before using them
3. **Check model availability** before initialization
4. **Provide meaningful error messages** with context
5. **Log errors** for debugging purposes

## Usage Examples

### Basic Pipeline Setup

```python
from src.core.rag_pipeline import RAGPipeline

config = {
    "chunker": {"type": "line_chunker", "chunk_size": 1000},
    "embedding": {"type": "openai", "model": "text-embedding-3-small"},
    "vector_store": {"type": "in_memory", "similarity_metric": "cosine"},
    "language_model": {"type": "openai", "model": "gpt-4o-mini"}
}

pipeline = RAGPipeline(config)
pipeline.index_documents(["Document 1", "Document 2"])
answer = pipeline.query("What is the main topic?")
```

### Running Experiments

```python
from src.core.experiment_runner import ExperimentRunner
from src.evaluations import RetrievalEvaluator, GenerationEvaluator

evaluators = {
    "retrieval": RetrievalEvaluator(),
    "generation": GenerationEvaluator()
}

runner = ExperimentRunner(evaluators)
results = runner.run_experiment(config, qa_pairs, "test_experiment")
```

### Component Registration

```python
from src.core.component_loader import ComponentLoader
from src.components.chunkers import CustomChunker

loader = ComponentLoader()
loader.register_component("chunker", "custom", CustomChunker)
chunker = loader.load_component("chunker", {"type": "custom", "param": "value"})
```

For more examples and tutorials, see the [ResearchRAG.ipynb](../src/ResearchRAG.ipynb) notebook.