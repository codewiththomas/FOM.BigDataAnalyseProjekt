import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
        "Machine learning (ML) is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.",
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
        "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
        "Big data refers to data sets that are too large or complex to be dealt with by traditional data-processing application software.",
        "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user."
    ]


@pytest.fixture
def sample_queries():
    """Provide sample queries for testing."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is natural language processing?",
        "Explain deep learning",
        "What is computer vision?",
        "How does reinforcement learning work?",
        "What are neural networks?",
        "What is data science?",
        "What is big data?",
        "What is cloud computing?"
    ]


@pytest.fixture
def sample_ground_truth():
    """Provide ground truth answers for testing."""
    return [
        "Artificial intelligence is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals.",
        "Machine learning allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.",
        "Natural language processing is a subfield concerned with interactions between computers and human language.",
        "Deep learning is part of machine learning methods based on artificial neural networks with representation learning.",
        "Computer vision is a field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Reinforcement learning is concerned with how intelligent agents should take actions in an environment to maximize cumulative reward.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
        "Data science uses scientific methods to extract knowledge and insights from structured and unstructured data.",
        "Big data refers to data sets that are too large or complex for traditional data-processing applications.",
        "Cloud computing is the on-demand availability of computer system resources without direct active management by the user."
    ]


@pytest.fixture
def basic_rag_config():
    """Provide basic RAG configuration for testing."""
    return {
        "chunker": {
            "type": "line_chunker",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "embedding": {
            "type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2"
        },
        "vector_store": {
            "type": "in_memory"
        },
        "language_model": {
            "type": "ollama",
            "model_name": "llama3.2"
        }
    }


@pytest.fixture
def alternative_rag_configs():
    """Provide alternative RAG configurations for testing."""
    return {
        "recursive_chunking": {
            "chunker": {
                "type": "recursive_chunker",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "in_memory"
            },
            "language_model": {
                "type": "ollama",
                "model_name": "llama3.2"
            }
        },
        "semantic_chunking": {
            "chunker": {
                "type": "semantic_chunker",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "in_memory"
            },
            "language_model": {
                "type": "ollama",
                "model_name": "llama3.2"
            }
        },
        "chroma_store": {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 500
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "chroma",
                "collection_name": "test_collection"
            },
            "language_model": {
                "type": "ollama",
                "model_name": "llama3.2"
            }
        },
        "faiss_store": {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 500
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "faiss",
                "embedding_dimension": 384
            },
            "language_model": {
                "type": "ollama",
                "model_name": "llama3.2"
            }
        }
    }


@pytest.fixture
def evaluators():
    """Provide evaluators for testing."""
    from evaluations import RetrievalEvaluator, GenerationEvaluator, PerformanceEvaluator, RAGEvaluator

    retrieval_eval = RetrievalEvaluator()
    generation_eval = GenerationEvaluator()
    performance_eval = PerformanceEvaluator()
    rag_eval = RAGEvaluator(retrieval_eval, generation_eval, performance_eval)

    return {
        "retrieval": retrieval_eval,
        "generation": generation_eval,
        "performance": performance_eval,
        "rag": rag_eval
    }


@pytest.fixture
def dependency_checker():
    """Check which dependencies are available."""
    dependencies = {
        "numpy": False,
        "sentence_transformers": False,
        "chromadb": False,
        "faiss": False,
        "openai": False,
        "ollama": False
    }

    # Check numpy
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass

    # Check sentence-transformers
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
    except ImportError:
        pass

    # Check chromadb
    try:
        import chromadb
        dependencies["chromadb"] = True
    except ImportError:
        pass

    # Check faiss
    try:
        import faiss
        dependencies["faiss"] = True
    except ImportError:
        pass

    # Check openai
    try:
        import openai
        dependencies["openai"] = True
    except ImportError:
        pass

    # Check ollama availability (basic check)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        dependencies["ollama"] = response.status_code == 200
    except:
        dependencies["ollama"] = False

    return dependencies


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Ensure test directories exist
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "..", "data")

    # Create necessary directories
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed", "chunks"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed", "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "evaluation", "results"), exist_ok=True)

    yield

    # Cleanup after test if needed
    # (Individual tests should handle their own cleanup)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama to be running"
    )
    config.addinivalue_line(
        "markers", "requires_openai: marks tests that require OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU acceleration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["benchmark", "performance", "large"]):
            item.add_marker(pytest.mark.slow)

        # Mark tests requiring external services
        if "ollama" in item.name.lower():
            item.add_marker(pytest.mark.requires_ollama)

        if "openai" in item.name.lower():
            item.add_marker(pytest.mark.requires_openai)


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Skip tests based on markers and available dependencies
    if item.get_closest_marker("requires_ollama"):
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            if response.status_code != 200:
                pytest.skip("Ollama service not available")
        except:
            pytest.skip("Ollama service not available")

    if item.get_closest_marker("requires_openai"):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

    if item.get_closest_marker("requires_gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available for GPU check")


# Custom pytest plugins
class DependencyPlugin:
    """Plugin to handle dependency-related test skipping."""

    def pytest_runtest_setup(self, item):
        """Skip tests based on missing dependencies."""
        # Check for required imports in test
        test_file = item.fspath
        with open(test_file, 'r') as f:
            content = f.read()

        # Skip if specific imports are missing
        if "import chromadb" in content:
            try:
                import chromadb
            except ImportError:
                pytest.skip("chromadb not available")

        if "import faiss" in content:
            try:
                import faiss
            except ImportError:
                pytest.skip("faiss not available")

        if "import sentence_transformers" in content:
            try:
                import sentence_transformers
            except ImportError:
                pytest.skip("sentence-transformers not available")


# Register the plugin
# pytest_plugins = [DependencyPlugin()]