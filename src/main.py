"""
Main orchestration script for the Simple Research RAG system.
"""

import os
import argparse
import yaml
from typing import Dict, Any, List
from pathlib import Path

from data_loader import DataLoader, DocumentChunker, create_sample_test_questions
from retriever import FAISSRetriever
from generator import get_generator, AVAILABLE_MODELS
from evaluator import RAGEvaluator


class SimpleRAG:
    """Main RAG system orchestrator."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RAG system with configuration."""
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize components
        self.data_loader = None
        self.retriever = None
        self.generator = None
        self.evaluator = None

        self._setup_components()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Creating default config...")
            return self._create_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        default_config = {
            "model": {
                "name": "gpt-4o-mini",
                "provider": "openai",
                "temperature": 0.1,
                "max_tokens": 1000
            },
            "retrieval": {
                "embedding_model": "text-embedding-3-small",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "top_k": 5,
                "similarity_threshold": 0.7
            },
            "evaluation": {
                "metrics": ["ragas", "bertscore", "performance"],
                "test_size": 50,
                "batch_size": 5
            },
            "data": {
                "documents_path": "data/documents",
                "test_questions_path": "data/test_questions.json",
                "index_path": "data/faiss_index"
            },
            "results": {
                "output_dir": "results"
            }
        }

        # Save default config
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            print(f"Default configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving default config: {e}")

        return default_config

    def _setup_components(self):
        """Initialize all RAG components."""
        # Data loader
        data_config = self.config.get("data", {})
        self.data_loader = DataLoader(
            documents_path=data_config.get("documents_path", "data/documents"),
            test_questions_path=data_config.get("test_questions_path", "data/test_questions.json")
        )

        # Document chunker
        retrieval_config = self.config.get("retrieval", {})
        chunker = DocumentChunker(
            chunk_size=retrieval_config.get("chunk_size", 500),
            chunk_overlap=retrieval_config.get("chunk_overlap", 100)
        )
        self.data_loader.set_chunker(chunker)

        # Retriever
        self.retriever = FAISSRetriever(
            embedding_model_name=retrieval_config.get("embedding_model", "text-embedding-3-small")
        )

        # Generator
        model_config = self.config.get("model", {})
        # Add API keys from environment
        model_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        model_config["groq_api_key"] = os.getenv("GROQ_API_KEY")

        self.generator = get_generator(model_config)

        # Evaluator
        self.evaluator = RAGEvaluator(self.config)

        print("All components initialized successfully")

    def setup_data(self, force_rebuild: bool = False):
        """Load documents and build index."""
        data_config = self.config.get("data", {})
        index_path = data_config.get("index_path", "data/faiss_index")

        # Try to load existing index
        if not force_rebuild and self.retriever.load_index(index_path):
            print("Loaded existing FAISS index")
            return

        print("Building new index...")

        # Load documents
        documents = self.data_loader.load_documents()
        if not documents:
            print("No documents found. Please add documents to the documents directory.")
            return

        # Chunk documents
        chunks = self.data_loader.chunk_documents(documents)

        # Build index
        self.retriever.build_index(chunks)

        # Save index
        self.retriever.save_index(index_path)

        print(f"Index built and saved to {index_path}")

    def setup_test_questions(self):
        """Setup test questions if they don't exist."""
        questions = self.data_loader.load_test_questions()

        if not questions:
            print("No test questions found. Creating sample questions...")
            sample_questions = create_sample_test_questions()
            self.data_loader.save_test_questions(sample_questions)
            questions = sample_questions

        return questions

    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Process a single query through the RAG pipeline."""
        # Load index if not already loaded
        if not self.retriever.chunks:
            data_config = self.config.get("data", {})
            index_path = data_config.get("index_path", "data/faiss_index")
            if not self.retriever.load_index(index_path):
                raise ValueError("Index not loaded and could not be loaded from disk. Run setup_data() first.")

        # Use config default if not specified
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)

        # Retrieve relevant contexts
        retrieved_chunks = self.retriever.search(
            query=question,
            top_k=top_k,
            similarity_threshold=self.config.get("retrieval", {}).get("similarity_threshold", 0.0)
        )

        # Extract context texts
        contexts = [chunk["text"] for chunk in retrieved_chunks]

        # Generate answer
        result = self.generator.generate(question, contexts)

        # Add retrieval info
        result["retrieved_contexts"] = contexts
        result["retrieval_scores"] = [chunk.get("similarity_score", 0) for chunk in retrieved_chunks]
        result["question"] = question

        return result

    def evaluate(self, test_size: int = None) -> Dict[str, Any]:
        """Run evaluation on test questions."""
        # Setup data if needed
        if not self.retriever.chunks:
            self.setup_data()

        # Load test questions
        questions = self.setup_test_questions()

        # Limit test size
        if test_size is None:
            test_size = self.config.get("evaluation", {}).get("test_size", len(questions))

        test_questions = questions[:min(test_size, len(questions))]

        print(f"Running evaluation on {len(test_questions)} questions...")

        # Process each question
        test_data = []

        for i, question_item in enumerate(test_questions):
            print(f"Processing question {i+1}/{len(test_questions)}: {question_item['question'][:80]}...")

            try:
                # Query RAG system
                result = self.query(question_item["question"])

                # Prepare evaluation data
                eval_item = {
                    "question": question_item["question"],
                    "generated_answer": result["response"],
                    "reference_answer": question_item["reference_answer"],
                    "retrieved_contexts": result["retrieved_contexts"],
                    "metadata": {
                        "model": result.get("model"),
                        "execution_time": result.get("execution_time"),
                        "tokens_per_second": result.get("tokens_per_second"),
                        "cost_estimate": result.get("cost_estimate", 0),
                        "usage": result.get("usage", {}),
                        "category": question_item.get("category", "unknown")
                    }
                }

                test_data.append(eval_item)

            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                continue

        # Run evaluation
        results = self.evaluator.evaluate_batch(test_data)

        # Save results
        model_name = self.config.get("model", {}).get("name", "unknown")
        output_dir = self.config.get("results", {}).get("output_dir", "results")

        saved_files = self.evaluator.save_results(results, output_dir, model_name)

        # Compute aggregates
        aggregates = self.evaluator.compute_aggregate_metrics(results)

        return {
            "results": results,
            "aggregates": aggregates,
            "saved_files": saved_files,
            "test_size": len(test_data),
            "model": model_name
        }

    def compare_models(self) -> None:
        """Compare results across different models."""
        output_dir = self.config.get("results", {}).get("output_dir", "results")
        comparison_df = self.evaluator.compare_models(output_dir)

        if not comparison_df.empty:
            print("\nModel Comparison:")
            print("=" * 50)
            print(comparison_df.to_string(index=False))

            # Save comparison
            comparison_file = Path(output_dir) / "model_comparison.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\nComparison saved to {comparison_file}")
        else:
            print("No evaluation results found for comparison.")

    def list_models(self) -> None:
        """List available models."""
        print("\nAvailable Models:")
        print("=" * 50)

        for model_id, model_info in AVAILABLE_MODELS.items():
            print(f"ID: {model_id}")
            print(f"  Name: {model_info['name']}")
            print(f"  Provider: {model_info['provider']}")
            print(f"  Description: {model_info['description']}")
            print()

    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        if model_name in AVAILABLE_MODELS:
            model_config = AVAILABLE_MODELS[model_name].copy()

            # Update config
            self.config["model"].update(model_config)

            # Add API keys
            model_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
            model_config["groq_api_key"] = os.getenv("GROQ_API_KEY")

            # Recreate generator
            self.generator = get_generator(model_config)

            print(f"Switched to model: {model_name}")

            # Save updated config
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                print(f"Configuration updated in {self.config_path}")
            except Exception as e:
                print(f"Error saving config: {e}")
        else:
            print(f"Model '{model_name}' not found. Available models:")
            self.list_models()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Research RAG System")

    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--setup", action="store_true", help="Setup data and build index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare model results")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", help="Switch to specified model")
    parser.add_argument("--query", help="Process a single query")
    parser.add_argument("--test-size", type=int, help="Number of test questions for evaluation")

    args = parser.parse_args()

    # Initialize RAG system
    try:
        rag = SimpleRAG(config_path=args.config)
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return

    # Execute commands
    try:
        if args.list_models:
            rag.list_models()

        elif args.model:
            rag.switch_model(args.model)

        elif args.setup or args.rebuild:
            rag.setup_data(force_rebuild=args.rebuild)

        elif args.query:
            print(f"Query: {args.query}")
            result = rag.query(args.query)
            print(f"\nAnswer: {result['response']}")
            print(f"Model: {result['model']}")
            print(f"Execution time: {result.get('execution_time', 0):.2f}s")
            print(f"Retrieved {len(result['retrieved_contexts'])} contexts")

        elif args.evaluate:
            print("Running evaluation...")
            eval_results = rag.evaluate(test_size=args.test_size)

            print(f"\nEvaluation completed!")
            print(f"Model: {eval_results['model']}")
            print(f"Test size: {eval_results['test_size']}")
            print(f"Results saved to: {eval_results['saved_files']['aggregates']}")

            # Print key metrics
            aggregates = eval_results['aggregates']
            print(f"\nKey Metrics:")
            for key, value in aggregates.items():
                if "mean" in key:
                    print(f"  {key}: {value:.3f}")

        elif args.compare:
            rag.compare_models()

        else:
            print("No action specified. Use --help for available options.")
            print("\nQuick start:")
            print("1. python src/main.py --setup")
            print("2. python src/main.py --query 'Was ist die DSGVO?'")
            print("3. python src/main.py --evaluate")

    except Exception as e:
        print(f"Error executing command: {e}")


if __name__ == "__main__":
    main()