import json
import time
from typing import List, Dict, Any
from pathlib import Path
import logging
from factory import RAGFactory
from dataset import DSGVODataset
from pipeline import RAGPipeline
from evaluation import EvaluationManager

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Main RAG evaluation orchestrator"""

    def __init__(self, config_path: str, dataset_path: str):
        self.factory = RAGFactory(config_path)
        self.dataset = DSGVODataset(dataset_path)
        self.pipeline = None
        self.evaluator = None
        self.results = []

        logger.info("RAG Evaluator initialized")

    def setup_pipeline(self):
        """Setup the RAG pipeline"""
        logger.info("Setting up RAG pipeline...")

        self.pipeline = self.factory.create_pipeline()
        self.evaluator = self.factory.create_evaluator()

        # Index documents
        documents = self.dataset.get_documents()
        self.pipeline.index_documents(documents)

        logger.info("Pipeline setup complete")

    def run_evaluation(self, num_qa: int = 50, save_results: bool = True) -> Dict[str, Any]:
        """Run complete evaluation on the RAG system"""
        if not self.pipeline:
            self.setup_pipeline()

        logger.info(f"Starting evaluation with {num_qa} QA pairs...")

        # Get evaluation subset
        qa_pairs = self.dataset.get_evaluation_subset(num_qa)

        # Run evaluation
        start_time = time.time()

        for i, qa_pair in enumerate(qa_pairs):
            logger.info(f"Evaluating QA pair {i+1}/{len(qa_pairs)}: {qa_pair['question'][:50]}...")

            try:
                # Query the pipeline
                result = self.pipeline.query(qa_pair['question'])

                # Evaluate the response
                evaluation = self.evaluator.evaluate_query(
                    query=qa_pair['question'],
                    ground_truth=qa_pair['ground_truth'],
                    response=result.response,
                    retrieved_chunks=result.chunks
                )

                # Combine results
                qa_result = {
                    'qa_id': qa_pair['id'],
                    'question': qa_pair['question'],
                    'ground_truth': qa_pair['ground_truth'],
                    'response': result.response,
                    'retrieved_chunks': len(result.chunks),
                    'query_time': result.metadata.get('query_time', 0.0),
                    'evaluation': evaluation,
                    'metadata': result.metadata
                }

                self.results.append(qa_result)

            except Exception as e:
                logger.error(f"Error evaluating QA pair {qa_pair['id']}: {e}")
                # Add error result
                error_result = {
                    'qa_id': qa_pair['id'],
                    'question': qa_pair['question'],
                    'ground_truth': qa_pair['ground_truth'],
                    'response': f"ERROR: {e}",
                    'retrieved_chunks': 0,
                    'query_time': 0.0,
                    'evaluation': {'error': str(e)},
                    'metadata': {}
                }
                self.results.append(error_result)

        total_time = time.time() - start_time

        # Calculate summary statistics
        summary = self._calculate_summary()
        summary['total_evaluation_time'] = total_time
        summary['qa_pairs_evaluated'] = len(qa_pairs)

        logger.info(f"Evaluation complete. Total time: {total_time:.2f}s")
        logger.info(f"Summary: {summary}")

        # Save results if requested
        if save_results:
            self._save_results(summary)

        return summary

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        if not self.results:
            return {}

        # Initialize metric accumulators
        metrics = {}
        for result in self.results:
            if 'evaluation' in result and isinstance(result['evaluation'], dict):
                for metric, value in result['evaluation'].items():
                    if isinstance(value, (int, float)) and value >= 0:  # Skip error indicators
                        if metric not in metrics:
                            metrics[metric] = []
                        metrics[metric].append(value)

        # Calculate averages
        summary = {}
        for metric, values in metrics.items():
            if values:
                summary[f'avg_{metric}'] = sum(values) / len(values)
                summary[f'min_{metric}'] = min(values)
                summary[f'max_{metric}'] = max(values)

        # Add pipeline info
        if self.pipeline:
            summary['pipeline_info'] = self.pipeline.get_pipeline_info()

        # Add evaluation info
        if self.evaluator:
            summary['evaluation_info'] = self.evaluator.get_evaluation_info()

        return summary

    def _save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {results_file} and {summary_file}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results"""
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary"""
        return self._calculate_summary()
