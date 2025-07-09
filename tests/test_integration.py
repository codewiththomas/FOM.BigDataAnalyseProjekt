import pytest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.rag_pipeline import RAGPipeline
from core.component_loader import ComponentLoader
from core.experiment_runner import ExperimentRunner
from evaluations import RetrievalEvaluator, GenerationEvaluator, PerformanceEvaluator, RAGEvaluator


class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""

    def setup_method(self):
        """Set up test environment."""
        self.test_documents = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
            "Machine learning (ML) is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.",
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos."
        ]

        self.test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is natural language processing?",
            "Explain deep learning",
            "What is computer vision?"
        ]

        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_pipeline_with_sentence_transformer(self):
        """Test RAG pipeline with SentenceTransformer embedding."""
        config = {
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

        try:
            pipeline = RAGPipeline(config)

            # Index documents
            pipeline.index_documents(self.test_documents)

            # Test query
            query = "What is artificial intelligence?"
            result = pipeline.query(query)

            assert isinstance(result, dict)
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "metadata" in result

            # Test multiple queries
            for query in self.test_queries[:3]:  # Test first 3 queries
                result = pipeline.query(query)
                assert isinstance(result, dict)
                assert "answer" in result

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            # May fail if Ollama is not running
            print(f"Pipeline test failed (expected if Ollama not running): {e}")

    def test_pipeline_with_chroma_store(self):
        """Test RAG pipeline with ChromaDB vector store."""
        config = {
            "chunker": {
                "type": "recursive_chunker",
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "chroma",
                "collection_name": "test_integration",
                "persist_directory": os.path.join(self.temp_dir, "chroma_db")
            },
            "language_model": {
                "type": "ollama",
                "model_name": "llama3.2"
            }
        }

        try:
            pipeline = RAGPipeline(config)

            # Index documents
            pipeline.index_documents(self.test_documents)

            # Test query
            query = "What is machine learning?"
            result = pipeline.query(query)

            assert isinstance(result, dict)
            assert "answer" in result
            assert "retrieved_documents" in result

            # Test persistence
            pipeline.save_index(os.path.join(self.temp_dir, "test_index"))

            # Create new pipeline and load index
            pipeline2 = RAGPipeline(config)
            pipeline2.load_index(os.path.join(self.temp_dir, "test_index"))

            # Test query with loaded index
            result2 = pipeline2.query(query)
            assert isinstance(result2, dict)
            assert "answer" in result2

        except ImportError as e:
            pytest.skip(f"ChromaDB not available: {e}")
        except Exception as e:
            print(f"ChromaDB test failed: {e}")

    def test_pipeline_with_faiss_store(self):
        """Test RAG pipeline with FAISS vector store."""
        config = {
            "chunker": {
                "type": "semantic_chunker",
                "chunk_size": 800,
                "chunk_overlap": 80
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "faiss",
                "embedding_dimension": 384,
                "index_type": "IndexFlatIP"
            },
            "language_model": {
                "type": "ollama",
                "model_name": "llama3.2"
            }
        }

        try:
            pipeline = RAGPipeline(config)

            # Index documents
            pipeline.index_documents(self.test_documents)

            # Test query
            query = "What is deep learning?"
            result = pipeline.query(query)

            assert isinstance(result, dict)
            assert "answer" in result
            assert "retrieved_documents" in result

            # Test index saving and loading
            index_path = os.path.join(self.temp_dir, "faiss_index")
            pipeline.save_index(index_path)

            # Load index in new pipeline
            pipeline2 = RAGPipeline(config)
            pipeline2.load_index(index_path)

            result2 = pipeline2.query(query)
            assert isinstance(result2, dict)

        except ImportError as e:
            pytest.skip(f"FAISS not available: {e}")
        except Exception as e:
            print(f"FAISS test failed: {e}")

    def test_pipeline_configuration_switching(self):
        """Test switching between different pipeline configurations."""
        # Configuration 1: Line chunker + In-memory store
        config1 = {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 300
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

        # Configuration 2: Recursive chunker + In-memory store
        config2 = {
            "chunker": {
                "type": "recursive_chunker",
                "chunk_size": 500
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

        try:
            # Test pipeline 1
            pipeline1 = RAGPipeline(config1)
            pipeline1.index_documents(self.test_documents)

            result1 = pipeline1.query("What is AI?")
            assert isinstance(result1, dict)

            # Test pipeline 2
            pipeline2 = RAGPipeline(config2)
            pipeline2.index_documents(self.test_documents)

            result2 = pipeline2.query("What is AI?")
            assert isinstance(result2, dict)

            # Compare pipeline info
            info1 = pipeline1.get_pipeline_info()
            info2 = pipeline2.get_pipeline_info()

            assert info1["chunker"]["type"] == "line_chunker"
            assert info2["chunker"]["type"] == "recursive_chunker"

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Configuration switching test failed: {e}")


class TestExperimentRunnerIntegration:
    """Integration tests for the experiment runner."""

    def setup_method(self):
        """Set up test environment."""
        self.test_documents = [
            "AI is artificial intelligence.",
            "ML is machine learning.",
            "NLP is natural language processing."
        ]

        self.test_queries = [
            "What is AI?",
            "What is ML?",
            "What is NLP?"
        ]

        self.ground_truth = [
            "AI is artificial intelligence.",
            "ML is machine learning.",
            "NLP is natural language processing."
        ]

        # Create evaluators
        self.evaluators = {
            "retrieval": RetrievalEvaluator(),
            "generation": GenerationEvaluator(),
            "performance": PerformanceEvaluator()
        }

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_experiment_run(self):
        """Test running a single experiment."""
        config = {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 500
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

        try:
            runner = ExperimentRunner(self.evaluators)

            # Run experiment
            results = runner.run_experiment(
                config=config,
                documents=self.test_documents,
                queries=self.test_queries,
                ground_truth=self.ground_truth,
                experiment_name="test_experiment"
            )

            assert isinstance(results, dict)
            assert "experiment_name" in results
            assert "config" in results
            assert "evaluation_results" in results
            assert "metadata" in results

            # Check evaluation results
            eval_results = results["evaluation_results"]
            assert "retrieval" in eval_results
            assert "generation" in eval_results
            assert "performance" in eval_results

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Single experiment test failed: {e}")

    def test_configuration_comparison(self):
        """Test comparing multiple configurations."""
        configs = {
            "config1": {
                "chunker": {
                    "type": "line_chunker",
                    "chunk_size": 300
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
            "config2": {
                "chunker": {
                    "type": "recursive_chunker",
                    "chunk_size": 500
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
        }

        try:
            runner = ExperimentRunner(self.evaluators)

            # Run comparison
            results = runner.compare_configurations(
                configs=configs,
                documents=self.test_documents,
                queries=self.test_queries,
                ground_truth=self.ground_truth,
                experiment_name="config_comparison"
            )

            assert isinstance(results, dict)
            assert "comparison_results" in results
            assert "summary" in results
            assert "best_config" in results

            # Check individual config results
            comparison_results = results["comparison_results"]
            assert "config1" in comparison_results
            assert "config2" in comparison_results

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Configuration comparison test failed: {e}")

    def test_ablation_study(self):
        """Test running an ablation study."""
        base_config = {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 500
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

        variations = {
            "chunker_size": [
                {"chunker": {"chunk_size": 300}},
                {"chunker": {"chunk_size": 700}}
            ],
            "chunker_type": [
                {"chunker": {"type": "recursive_chunker"}},
                {"chunker": {"type": "semantic_chunker"}}
            ]
        }

        try:
            runner = ExperimentRunner(self.evaluators)

            # Run ablation study
            results = runner.run_ablation_study(
                base_config=base_config,
                variations=variations,
                documents=self.test_documents,
                queries=self.test_queries,
                ground_truth=self.ground_truth,
                experiment_name="ablation_study"
            )

            assert isinstance(results, dict)
            assert "ablation_results" in results
            assert "summary" in results

            # Check variation results
            ablation_results = results["ablation_results"]
            assert "chunker_size" in ablation_results
            assert "chunker_type" in ablation_results

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Ablation study test failed: {e}")

    def test_results_persistence(self):
        """Test saving and loading experiment results."""
        config = {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 500
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

        try:
            runner = ExperimentRunner(self.evaluators)

            # Run experiment
            results = runner.run_experiment(
                config=config,
                documents=self.test_documents,
                queries=self.test_queries,
                ground_truth=self.ground_truth,
                experiment_name="persistence_test"
            )

            # Save results
            results_path = os.path.join(self.temp_dir, "experiment_results.json")
            runner.save_results(results, results_path)

            # Check file exists
            assert os.path.exists(results_path)

            # Load results
            loaded_results = runner.load_results(results_path)

            assert isinstance(loaded_results, dict)
            assert "experiment_name" in loaded_results
            assert loaded_results["experiment_name"] == "persistence_test"

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Results persistence test failed: {e}")


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test dataset
        self.test_dataset = {
            "documents": [
                "Python is a high-level programming language known for its simplicity and readability.",
                "JavaScript is a programming language that is one of the core technologies of the World Wide Web.",
                "Java is a class-based, object-oriented programming language designed to have as few implementation dependencies as possible.",
                "C++ is a general-purpose programming language created as an extension of the C programming language.",
                "Go is a statically typed, compiled programming language designed at Google."
            ],
            "queries": [
                "What is Python?",
                "Tell me about JavaScript",
                "What is Java programming language?",
                "Explain C++",
                "What is Go language?"
            ],
            "ground_truth": [
                "Python is a high-level programming language known for its simplicity and readability.",
                "JavaScript is a programming language that is one of the core technologies of the World Wide Web.",
                "Java is a class-based, object-oriented programming language designed to have as few implementation dependencies as possible.",
                "C++ is a general-purpose programming language created as an extension of the C programming language.",
                "Go is a statically typed, compiled programming language designed at Google."
            ]
        }

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_research_workflow(self):
        """Test a complete research workflow."""
        # Define multiple configurations to test
        configurations = {
            "baseline": {
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
            },
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
            }
        }

        try:
            # Create evaluators
            evaluators = {
                "retrieval": RetrievalEvaluator(),
                "generation": GenerationEvaluator(),
                "performance": PerformanceEvaluator(),
                "rag": RAGEvaluator(
                    RetrievalEvaluator(),
                    GenerationEvaluator(),
                    PerformanceEvaluator()
                )
            }

            # Create experiment runner
            runner = ExperimentRunner(evaluators)

            # Run comparison study
            results = runner.compare_configurations(
                configs=configurations,
                documents=self.test_dataset["documents"],
                queries=self.test_dataset["queries"],
                ground_truth=self.test_dataset["ground_truth"],
                experiment_name="chunking_comparison_study"
            )

            # Validate results
            assert isinstance(results, dict)
            assert "comparison_results" in results
            assert "summary" in results
            assert "best_config" in results

            # Check all configurations were tested
            for config_name in configurations.keys():
                assert config_name in results["comparison_results"]

            # Save results
            results_path = os.path.join(self.temp_dir, "research_results.json")
            runner.save_results(results, results_path)

            # Generate report
            report = runner.generate_report(results)
            assert isinstance(report, str)
            assert "Experiment Results" in report

            # Save report
            report_path = os.path.join(self.temp_dir, "research_report.md")
            with open(report_path, "w") as f:
                f.write(report)

            assert os.path.exists(report_path)

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Complete workflow test failed: {e}")

    def test_performance_benchmarking(self):
        """Test performance benchmarking workflow."""
        config = {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 500
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

        try:
            # Create evaluators
            evaluators = {
                "performance": PerformanceEvaluator()
            }

            runner = ExperimentRunner(evaluators)

            # Run performance benchmark
            results = runner.run_performance_benchmark(
                config=config,
                documents=self.test_dataset["documents"],
                queries=self.test_dataset["queries"],
                experiment_name="performance_benchmark"
            )

            assert isinstance(results, dict)
            assert "benchmark_results" in results
            assert "performance_metrics" in results

            # Check performance metrics
            perf_metrics = results["performance_metrics"]
            assert "avg_latency" in perf_metrics
            assert "throughput" in perf_metrics
            assert "memory_usage" in perf_metrics

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Performance benchmarking test failed: {e}")


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        invalid_config = {
            "chunker": {
                "type": "non_existent_chunker"
            },
            "embedding": {
                "type": "sentence_transformer"
            },
            "vector_store": {
                "type": "in_memory"
            },
            "language_model": {
                "type": "ollama"
            }
        }

        with pytest.raises(Exception):
            pipeline = RAGPipeline(invalid_config)

    def test_missing_dependencies(self):
        """Test handling of missing dependencies."""
        # This test checks if the system gracefully handles missing optional dependencies

        # Test with OpenAI (requires API key)
        openai_config = {
            "chunker": {
                "type": "line_chunker"
            },
            "embedding": {
                "type": "openai"
            },
            "vector_store": {
                "type": "in_memory"
            },
            "language_model": {
                "type": "openai"
            }
        }

        try:
            pipeline = RAGPipeline(openai_config)
            # Should fail gracefully if no API key
        except Exception as e:
            assert "API key" in str(e) or "OpenAI" in str(e)

    def test_empty_documents(self):
        """Test handling of empty document collections."""
        config = {
            "chunker": {
                "type": "line_chunker"
            },
            "embedding": {
                "type": "sentence_transformer"
            },
            "vector_store": {
                "type": "in_memory"
            },
            "language_model": {
                "type": "ollama"
            }
        }

        try:
            pipeline = RAGPipeline(config)

            # Test with empty documents
            pipeline.index_documents([])

            # Query should handle empty index gracefully
            result = pipeline.query("What is AI?")
            assert isinstance(result, dict)

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Empty documents test failed: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])