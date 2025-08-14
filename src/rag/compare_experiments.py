#!/usr/bin/env python3
"""
ResearchRAG - Experiment Comparison Script
Compare multiple RAG configurations and generate comprehensive reports
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded successfully")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")

from evaluator import RAGEvaluator


class ExperimentComparator:
    """Compare multiple RAG experiment configurations"""

    def __init__(self, configs: List[str], dataset_path: str, output_dir: str = "comparison_results"):
        self.configs = configs
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = {}
        self.comparison_data = []

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized ExperimentComparator with {len(configs)} configurations")

    def run_all_experiments(self, num_qa: int = 50) -> Dict[str, Any]:
        """Run all experiments and collect results"""
        self.logger.info(f"Starting comparison of {len(self.configs)} configurations with {num_qa} QA pairs each")

        start_time = time.time()

        for config_path in self.configs:
            try:
                self.logger.info(f"Running experiment: {config_path}")

                # Extract experiment name
                experiment_name = Path(config_path).stem

                # Run evaluation
                evaluator = RAGEvaluator(config_path, self.dataset_path)
                summary = evaluator.run_evaluation(num_qa=num_qa, save_results=True)

                # Store results
                self.results[experiment_name] = summary

                # Add to comparison data
                self.comparison_data.append({
                    'experiment': experiment_name,
                    'config_file': config_path,
                    **self._extract_metrics(summary)
                })

                self.logger.info(f"✅ Completed experiment: {experiment_name}")

            except Exception as e:
                self.logger.error(f"❌ Failed experiment {config_path}: {e}")
                # Add failed experiment to comparison data
                self.comparison_data.append({
                    'experiment': Path(config_path).stem,
                    'config_file': config_path,
                    'status': 'FAILED',
                    'error': str(e)
                })

        total_time = time.time() - start_time
        self.logger.info(f"All experiments completed in {total_time:.2f}s")

        return self.results

    def _extract_metrics(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from evaluation summary"""
        metrics = {
            'status': 'SUCCESS',
            'total_evaluation_time': summary.get('total_evaluation_time', 0),
            'qa_pairs_evaluated': summary.get('qa_pairs_evaluated', 0)
        }

        # Core metrics
        core_metrics = ['avg_precision', 'avg_recall', 'avg_f1', 'avg_query_time']
        for metric in core_metrics:
            metrics[metric] = summary.get(metric, 0)

        # RAGAS metrics
        ragas_metrics = ['avg_faithfulness', 'avg_answer_relevance', 'avg_context_relevance', 'avg_context_utilization']
        for metric in ragas_metrics:
            metrics[metric] = summary.get(metric, 0)

        # Performance metrics
        perf_metrics = ['avg_tokens_per_second', 'avg_response_length', 'avg_word_count']
        for metric in perf_metrics:
            metrics[metric] = summary.get(metric, 0)

        # Pipeline info
        pipeline_info = summary.get('pipeline_info', {})
        metrics['llm_model'] = pipeline_info.get('llm', {}).get('name', 'unknown')
        metrics['embedding_model'] = pipeline_info.get('embedding', {}).get('name', 'unknown')
        metrics['chunking_strategy'] = pipeline_info.get('chunking', {}).get('name', 'unknown')
        metrics['retrieval_method'] = pipeline_info.get('retrieval', {}).get('name', 'unknown')

        return metrics

    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report"""
        if not self.comparison_data:
            return "No comparison data available"

        # Create DataFrame for analysis
        df = pd.DataFrame(self.comparison_data)

        # Filter successful experiments
        successful_df = df[df['status'] == 'SUCCESS'].copy()

        if successful_df.empty:
            return "No successful experiments to compare"

        # Generate report
        report = []
        report.append("=" * 80)
        report.append("RAG EXPERIMENT COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Configurations: {len(self.configs)}")
        report.append(f"Successful Experiments: {len(successful_df)}")
        report.append(f"Failed Experiments: {len(df) - len(successful_df)}")
        report.append("")

        # Overall performance ranking
        if len(successful_df) > 1:
            report.append("🏆 PERFORMANCE RANKINGS")
            report.append("-" * 40)

            # F1 Score ranking
            f1_ranking = successful_df.nlargest(3, 'avg_f1')[['experiment', 'avg_f1']]
            report.append("Top 3 by F1 Score:")
            for _, row in f1_ranking.iterrows():
                report.append(f"  {row['experiment']}: {row['avg_f1']:.4f}")
            report.append("")

            # Speed ranking
            speed_ranking = successful_df.nsmallest(3, 'avg_query_time')[['experiment', 'avg_query_time']]
            report.append("Top 3 by Speed (lowest query time):")
            for _, row in speed_ranking.iterrows():
                report.append(f"  {row['experiment']}: {row['avg_query_time']:.4f}s")
            report.append("")

            # Faithfulness ranking
            faithfulness_ranking = successful_df.nlargest(3, 'avg_faithfulness')[['experiment', 'avg_faithfulness']]
            report.append("Top 3 by Faithfulness:")
            for _, row in faithfulness_ranking.iterrows():
                report.append(f"  {row['experiment']}: {row['avg_faithfulness']:.4f}")
            report.append("")

        # Detailed metrics comparison
        report.append("📊 DETAILED METRICS COMPARISON")
        report.append("-" * 40)

        # Core metrics
        core_metrics = ['avg_precision', 'avg_recall', 'avg_f1']
        for metric in core_metrics:
            if metric in successful_df.columns:
                best_exp = successful_df.loc[successful_df[metric].idxmax()]
                report.append(f"Best {metric}: {best_exp['experiment']} ({best_exp[metric]:.4f})")

        report.append("")

        # RAGAS metrics
        ragas_metrics = ['avg_faithfulness', 'avg_answer_relevance', 'avg_context_relevance', 'avg_context_utilization']
        for metric in ragas_metrics:
            if metric in successful_df.columns:
                best_exp = successful_df.loc[successful_df[metric].idxmax()]
                report.append(f"Best {metric}: {best_exp['experiment']} ({best_exp[metric]:.4f})")

        report.append("")

        # Performance metrics
        perf_metrics = ['avg_query_time', 'avg_tokens_per_second']
        for metric in perf_metrics:
            if metric in successful_df.columns:
                if 'time' in metric:
                    best_exp = successful_df.loc[successful_df[metric].idxmin()]
                    report.append(f"Best {metric}: {best_exp['experiment']} ({best_exp[metric]:.4f}s)")
                else:
                    best_exp = successful_df.loc[successful_df[metric].idxmax()]
                    report.append(f"Best {metric}: {best_exp['experiment']} ({best_exp[metric]:.4f})")

        report.append("")

        # Configuration summary
        report.append("⚙️  CONFIGURATION SUMMARY")
        report.append("-" * 40)
        for _, row in successful_df.iterrows():
            report.append(f"Experiment: {row['experiment']}")
            report.append(f"  LLM: {row.get('llm_model', 'unknown')}")
            report.append(f"  Embedding: {row.get('embedding_model', 'unknown')}")
            report.append(f"  Chunking: {row.get('chunking_strategy', 'unknown')}")
            report.append(f"  Retrieval: {row.get('retrieval_method', 'unknown')}")
            report.append("")

        # Failed experiments
        failed_df = df[df['status'] == 'FAILED']
        if not failed_df.empty:
            report.append("❌ FAILED EXPERIMENTS")
            report.append("-" * 40)
            for _, row in failed_df.iterrows():
                report.append(f"Experiment: {row['experiment']}")
                report.append(f"Error: {row.get('error', 'Unknown error')}")
                report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def save_comparison_results(self):
        """Save comparison results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"comparison_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Save comparison data as CSV
        if self.comparison_data:
            df = pd.DataFrame(self.comparison_data)
            csv_file = self.output_dir / f"comparison_data_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

        # Save comparison report
        report = self.generate_comparison_report()
        report_file = self.output_dir / f"comparison_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # Print report to console
        print("\n" + report)

        self.logger.info(f"Comparison results saved to {self.output_dir}")
        self.logger.info(f"Files: {results_file.name}, {csv_file.name}, {report_file.name}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ResearchRAG - Compare multiple RAG configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline and Mixtral configurations
  python src/rag/compare_experiments.py --configs configs/baseline.yaml configs/mixtral_7b.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --num-qa 50

  # Compare with custom output directory
  python src/rag/compare_experiments.py --configs configs/*.yaml --dataset data/output/dsgvo_crawled_2025-08-14_1535.jsonl --output-dir my_comparison_results
        """
    )

    parser.add_argument(
        '--configs', '-c',
        nargs='+',
        required=True,
        help='Paths to YAML configuration files to compare'
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
        help='Number of QA pairs to evaluate per configuration (default: 50)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='comparison_results',
        help='Output directory for comparison results (default: comparison_results)'
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
    for config_path in args.configs:
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            return 1

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1

    try:
        # Initialize and run comparison
        print("🚀 Initializing ResearchRAG experiment comparison...")

        comparator = ExperimentComparator(
            configs=args.configs,
            dataset_path=args.dataset,
            output_dir=args.output_dir
        )

        print(f"📊 Running {len(args.configs)} experiments with {args.num_qa} QA pairs each...")

        # Run all experiments
        results = comparator.run_all_experiments(num_qa=args.num_qa)

        # Generate and save comparison
        comparator.save_comparison_results()

        print("✅ Experiment comparison complete!")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        logging.error("Comparison failed", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
