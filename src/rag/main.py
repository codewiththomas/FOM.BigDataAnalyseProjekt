#!/usr/bin/env python3
"""
ResearchRAG - Main CLI interface for RAG evaluation
"""

import argparse
import logging
import sys
from pathlib import Path
from .evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ResearchRAG - Modular RAG evaluation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default config
  python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl

  # Run evaluation with custom number of QA pairs
  python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --num-qa 100

  # Run evaluation without saving results
  python -m rag.main --config configs/baseline.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --no-save
        """
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Path to DSGVO dataset JSONL file'
    )

    parser.add_argument(
        '--num-qa', '-n',
        type=int,
        default=50,
        help='Number of QA pairs to evaluate (default: 50)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation results to files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate file paths
    config_path = Path(args.config)
    dataset_path = Path(args.dataset)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    try:
        # Initialize and run evaluation
        print("🚀 Initializing ResearchRAG evaluation...")

        evaluator = RAGEvaluator(
            config_path=str(config_path),
            dataset_path=str(dataset_path)
        )

        print(f"📊 Running evaluation on {args.num_qa} QA pairs...")

        summary = evaluator.run_evaluation(
            num_qa=args.num_qa,
            save_results=not args.no_save
        )

        # Print summary
        print("\n" + "="*50)
        print("📈 EVALUATION SUMMARY")
        print("="*50)

        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print("="*50)
        print("✅ Evaluation complete!")

        if not args.no_save:
            print("💾 Results saved to JSON files")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        logging.error("Evaluation failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
