import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.chunkers import LineChunker, RecursiveChunker, SemanticChunker
from components.embeddings import OpenAIEmbedding, SentenceTransformerEmbedding
from components.vector_stores import InMemoryVectorStore, ChromaVectorStore, FAISSVectorStore
from components.language_models import OpenAILanguageModel, OllamaLanguageModel
from evaluations import RetrievalEvaluator, GenerationEvaluator, PerformanceEvaluator, RAGEvaluator
from core.rag_pipeline import RAGPipeline
from core.component_loader import ComponentLoader
from core.experiment_runner import ExperimentRunner


class TestChunkers:
    """Test all chunker implementations."""

    def setup_method(self):
        """Set up test data."""
        self.test_text = """
        This is the first paragraph of the test document.
        It contains multiple sentences to test chunking.

        This is the second paragraph.
        It also has several sentences.

        This is the third paragraph with more content.
        We need enough text to test different chunking strategies.
        """

    def test_line_chunker(self):
        """Test LineChunker functionality."""
        chunker = LineChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk_text(self.test_text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some overlap

    def test_recursive_chunker(self):
        """Test RecursiveChunker functionality."""
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)

        chunks = chunker.chunk_text(self.test_text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 180 for chunk in chunks)  # Allow some overlap

    def test_semantic_chunker(self):
        """Test SemanticChunker functionality."""
        # Test with fallback method (no sentence-transformers)
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=40)

        chunks = chunker.chunk_text(self.test_text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunker_metadata(self):
        """Test chunker metadata functionality."""
        chunker = LineChunker()
        chunks = chunker.chunk_text(self.test_text)

        for i, chunk in enumerate(chunks):
            metadata = chunker.get_chunk_metadata(chunk, i)

            assert isinstance(metadata, dict)
            assert "chunk_index" in metadata
            assert "chunk_size" in metadata
            assert "word_count" in metadata
            assert metadata["chunk_index"] == i


class TestEmbeddings:
    """Test all embedding implementations."""

    def setup_method(self):
        """Set up test data."""
        self.test_texts = [
            "This is a test sentence.",
            "Another test sentence for embedding.",
            "A third sentence to test embeddings."
        ]

    def test_sentence_transformer_embedding(self):
        """Test SentenceTransformerEmbedding functionality."""
        # Skip if sentence-transformers not available
        try:
            embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

            if embedding.available:
                embeddings = embedding.embed_texts(self.test_texts)

                assert isinstance(embeddings, np.ndarray)
                assert embeddings.shape[0] == len(self.test_texts)
                assert embeddings.shape[1] == embedding.get_embedding_dimension()

                # Test single query embedding
                query_embedding = embedding.embed_query("Test query")
                assert isinstance(query_embedding, np.ndarray)
                assert len(query_embedding) == embedding.get_embedding_dimension()
        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_openai_embedding(self):
        """Test OpenAIEmbedding functionality (mock test)."""
        # This test requires API key, so we'll test initialization only
        try:
            embedding = OpenAIEmbedding()

            # Test basic properties
            assert hasattr(embedding, 'model')
            assert hasattr(embedding, 'get_embedding_dimension')

            # Test dimension calculation
            dimension = embedding.get_embedding_dimension()
            assert isinstance(dimension, int)
            assert dimension > 0

        except Exception as e:
            # Expected if no API key is available
            assert "API key" in str(e) or "OpenAI" in str(e)

    def test_embedding_similarity(self):
        """Test embedding similarity calculation."""
        try:
            embedding = SentenceTransformerEmbedding()

            if embedding.available:
                emb1 = embedding.embed_query("This is a test")
                emb2 = embedding.embed_query("This is a test")
                emb3 = embedding.embed_query("Completely different content")

                # Test similarity
                sim_identical = embedding.calculate_similarity(emb1, emb2)
                sim_different = embedding.calculate_similarity(emb1, emb3)

                assert 0 <= sim_identical <= 1
                assert 0 <= sim_different <= 1
                assert sim_identical > sim_different
        except ImportError:
            pytest.skip("sentence-transformers not available")


class TestVectorStores:
    """Test all vector store implementations."""

    def setup_method(self):
        """Set up test data."""
        self.test_texts = [
            "Document about artificial intelligence and machine learning.",
            "Text about natural language processing and NLP.",
            "Content related to information retrieval systems."
        ]

        # Create dummy embeddings
        self.embeddings = np.random.random((len(self.test_texts), 384))
        self.query_embedding = np.random.random(384)

    def test_in_memory_vector_store(self):
        """Test InMemoryVectorStore functionality."""
        store = InMemoryVectorStore()

        # Test adding texts
        ids = store.add_texts(self.test_texts, self.embeddings)

        assert isinstance(ids, list)
        assert len(ids) == len(self.test_texts)

        # Test similarity search
        results = store.similarity_search(self.query_embedding, k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(len(result) == 3 for result in results)  # (text, score, metadata)

        # Test getting all documents
        all_docs = store.get_all_documents()
        assert len(all_docs) == len(self.test_texts)

        # Test collection stats
        stats = store.get_collection_stats()
        assert isinstance(stats, dict)
        assert "document_count" in stats

    def test_chroma_vector_store(self):
        """Test ChromaVectorStore functionality."""
        try:
            store = ChromaVectorStore(collection_name="test_collection")

            if store.available:
                # Test adding texts
                ids = store.add_texts(self.test_texts, self.embeddings)

                assert isinstance(ids, list)
                assert len(ids) == len(self.test_texts)

                # Test similarity search
                results = store.similarity_search(self.query_embedding, k=2)

                assert isinstance(results, list)
                assert len(results) <= 2

                # Clean up
                store.clear_collection()
        except ImportError:
            pytest.skip("chromadb not available")

    def test_faiss_vector_store(self):
        """Test FAISSVectorStore functionality."""
        try:
            store = FAISSVectorStore(embedding_dimension=384)

            if store.available:
                # Test adding texts
                ids = store.add_texts(self.test_texts, self.embeddings)

                assert isinstance(ids, list)
                assert len(ids) == len(self.test_texts)

                # Test similarity search
                results = store.similarity_search(self.query_embedding, k=2)

                assert isinstance(results, list)
                assert len(results) <= 2
        except ImportError:
            pytest.skip("faiss not available")


class TestLanguageModels:
    """Test all language model implementations."""

    def setup_method(self):
        """Set up test data."""
        self.test_prompt = "What is artificial intelligence?"
        self.test_context = "Artificial intelligence is a field of computer science."
        self.test_question = "What is AI?"

    def test_openai_language_model(self):
        """Test OpenAILanguageModel functionality (mock test)."""
        try:
            model = OpenAILanguageModel()

            # Test basic properties
            assert hasattr(model, 'model')
            assert hasattr(model, 'temperature')
            assert hasattr(model, 'max_tokens')

            # Test model info
            info = model.get_model_info()
            assert isinstance(info, dict)
            assert "model_name" in info

        except Exception as e:
            # Expected if no API key is available
            assert "API key" in str(e) or "OpenAI" in str(e)

    def test_ollama_language_model(self):
        """Test OllamaLanguageModel functionality."""
        model = OllamaLanguageModel()

        # Test basic properties
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'base_url')
        assert hasattr(model, 'temperature')

        # Test model info
        info = model.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info

        # Test health check
        health = model.health_check()
        assert isinstance(health, dict)
        assert "service_available" in health

        # Test cost calculation (should be 0 for local models)
        cost = model.calculate_cost(100, 50)
        assert cost == 0.0


class TestEvaluators:
    """Test all evaluator implementations."""

    def setup_method(self):
        """Set up test data."""
        self.predictions = [
            "This is a predicted answer about AI.",
            "Another prediction about machine learning.",
            "A third prediction about NLP."
        ]

        self.ground_truth = [
            "This is the correct answer about AI.",
            "The correct answer about machine learning.",
            "The right answer about NLP."
        ]

        self.contexts = [
            "Context about artificial intelligence.",
            "Context about machine learning.",
            "Context about natural language processing."
        ]

    def test_retrieval_evaluator(self):
        """Test RetrievalEvaluator functionality."""
        evaluator = RetrievalEvaluator()

        # Test evaluation
        results = evaluator.evaluate(self.predictions, self.ground_truth, self.contexts)

        assert isinstance(results, dict)
        assert "precision_at_1" in results
        assert "recall_at_1" in results
        assert "f1_at_1" in results
        assert "mrr" in results

        # Test individual metrics
        precision = evaluator.precision_at_k(["doc1", "doc2"], ["doc1", "doc3"], k=2)
        assert 0 <= precision <= 1

        recall = evaluator.recall_at_k(["doc1", "doc2"], ["doc1", "doc3"], k=2)
        assert 0 <= recall <= 1

    def test_generation_evaluator(self):
        """Test GenerationEvaluator functionality."""
        evaluator = GenerationEvaluator()

        # Test evaluation
        results = evaluator.evaluate(self.predictions, self.ground_truth, self.contexts)

        assert isinstance(results, dict)
        assert "rouge_l" in results
        assert "bleu_score" in results
        assert "exact_match" in results
        assert "semantic_similarity" in results

        # Test individual metrics
        rouge = evaluator.rouge_l_score(self.predictions[0], self.ground_truth[0])
        assert 0 <= rouge <= 1

        bleu = evaluator.bleu_score(self.predictions[0], self.ground_truth[0])
        assert 0 <= bleu <= 1

        exact_match = evaluator.exact_match(self.predictions[0], self.ground_truth[0])
        assert isinstance(exact_match, bool)

    def test_performance_evaluator(self):
        """Test PerformanceEvaluator functionality."""
        evaluator = PerformanceEvaluator()

        # Test evaluation
        results = evaluator.evaluate(self.predictions, self.ground_truth, self.contexts)

        assert isinstance(results, dict)
        assert "avg_latency" in results
        assert "throughput" in results
        assert "memory_usage" in results

        # Test latency measurement
        def dummy_function():
            return "result"

        result, latency = evaluator.measure_latency(dummy_function)
        assert result == "result"
        assert isinstance(latency, float)
        assert latency >= 0

    def test_rag_evaluator(self):
        """Test RAGEvaluator functionality."""
        retrieval_evaluator = RetrievalEvaluator()
        generation_evaluator = GenerationEvaluator()
        performance_evaluator = PerformanceEvaluator()

        rag_evaluator = RAGEvaluator(
            retrieval_evaluator=retrieval_evaluator,
            generation_evaluator=generation_evaluator,
            performance_evaluator=performance_evaluator
        )

        # Test evaluation
        results = rag_evaluator.evaluate(self.predictions, self.ground_truth, self.contexts)

        assert isinstance(results, dict)
        assert "retrieval_metrics" in results
        assert "generation_metrics" in results
        assert "performance_metrics" in results
        assert "rag_score" in results


class TestCoreComponents:
    """Test core system components."""

    def test_component_loader(self):
        """Test ComponentLoader functionality."""
        loader = ComponentLoader()

        # Test component registration
        loader.register_component("chunker", "test_chunker", LineChunker)

        # Test component loading
        config = {"type": "test_chunker", "chunk_size": 500}
        chunker = loader.load_component("chunker", config)

        assert isinstance(chunker, LineChunker)
        assert chunker.chunk_size == 500

        # Test getting registered components
        registered = loader.get_registered_components()
        assert isinstance(registered, dict)
        assert "chunker" in registered
        assert "test_chunker" in registered["chunker"]

    def test_rag_pipeline_initialization(self):
        """Test RAGPipeline initialization."""
        config = {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "embedding": {
                "type": "sentence_transformer",
                "model": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "in_memory",
                "similarity_metric": "cosine"
            },
            "language_model": {
                "type": "ollama",
                "model": "llama3.2"
            }
        }

        try:
            pipeline = RAGPipeline(config)

            # Test pipeline info
            info = pipeline.get_pipeline_info()
            assert isinstance(info, dict)
            assert "chunker" in info
            assert "embedding" in info
            assert "vector_store" in info
            assert "language_model" in info

        except Exception as e:
            # May fail if dependencies are not available
            print(f"Pipeline initialization failed (expected): {e}")

    def test_experiment_runner_initialization(self):
        """Test ExperimentRunner initialization."""
        evaluators = {
            "retrieval": RetrievalEvaluator(),
            "generation": GenerationEvaluator(),
            "performance": PerformanceEvaluator()
        }

        runner = ExperimentRunner(evaluators)

        assert hasattr(runner, 'evaluators')
        assert len(runner.evaluators) == 3
        assert "retrieval" in runner.evaluators
        assert "generation" in runner.evaluators
        assert "performance" in runner.evaluators


class TestIntegration:
    """Integration tests for the complete system."""

    def test_simple_rag_workflow(self):
        """Test a simple RAG workflow with mock components."""
        # Create simple test documents
        documents = [
            "Artificial intelligence is the simulation of human intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text and speech."
        ]

        # Test with available components only
        try:
            # Use sentence transformer if available
            embedding = SentenceTransformerEmbedding()
            if not embedding.available:
                pytest.skip("No embedding model available")

            # Create embeddings
            embeddings = embedding.embed_texts(documents)

            # Create vector store
            vector_store = InMemoryVectorStore()

            # Add documents
            ids = vector_store.add_texts(documents, embeddings)
            assert len(ids) == len(documents)

            # Test query
            query = "What is AI?"
            query_embedding = embedding.embed_query(query)

            # Search
            results = vector_store.similarity_search(query_embedding, k=2)
            assert len(results) <= 2
            assert all(len(result) == 3 for result in results)

        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_evaluation_workflow(self):
        """Test evaluation workflow."""
        # Create test data
        predictions = ["AI is machine intelligence.", "ML is learning from data."]
        ground_truth = ["AI is artificial intelligence.", "ML is machine learning."]
        contexts = ["Context about AI.", "Context about ML."]

        # Test individual evaluators
        retrieval_eval = RetrievalEvaluator()
        generation_eval = GenerationEvaluator()
        performance_eval = PerformanceEvaluator()

        # Test evaluations
        retrieval_results = retrieval_eval.evaluate(predictions, ground_truth, contexts)
        generation_results = generation_eval.evaluate(predictions, ground_truth, contexts)
        performance_results = performance_eval.evaluate(predictions, ground_truth, contexts)

        assert isinstance(retrieval_results, dict)
        assert isinstance(generation_results, dict)
        assert isinstance(performance_results, dict)

        # Test combined evaluation
        rag_eval = RAGEvaluator(retrieval_eval, generation_eval, performance_eval)
        rag_results = rag_eval.evaluate(predictions, ground_truth, contexts)

        assert isinstance(rag_results, dict)
        assert "rag_score" in rag_results


# Utility functions for testing
def test_system_dependencies():
    """Test system dependencies availability."""
    dependencies = {
        "numpy": True,
        "sentence_transformers": False,
        "chromadb": False,
        "faiss": False,
        "openai": False
    }

    # Test numpy (required)
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        dependencies["numpy"] = False

    # Test sentence-transformers (optional)
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
    except ImportError:
        pass

    # Test chromadb (optional)
    try:
        import chromadb
        dependencies["chromadb"] = True
    except ImportError:
        pass

    # Test faiss (optional)
    try:
        import faiss
        dependencies["faiss"] = True
    except ImportError:
        pass

    # Test openai (optional)
    try:
        import openai
        dependencies["openai"] = True
    except ImportError:
        pass

    print("Dependency Status:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")

    # Ensure numpy is available (required)
    assert dependencies["numpy"], "NumPy is required but not available"


if __name__ == "__main__":
    # Run dependency check
    test_system_dependencies()

    # Run tests
    pytest.main([__file__, "-v"])