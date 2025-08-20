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
        dataset_config = self.factory.config.get_dataset_config()
        self.dataset = DSGVODataset(dataset_path, dataset_config)  # config hinzufügen
        self.pipeline = None
        self.evaluator = None
        self.results = []
        self.experiment_name = self.factory.experiment_name

        logger.info(f"RAG Evaluator initialized for experiment: {self.experiment_name}")

    def setup_pipeline(self):
        """Setup the RAG pipeline"""
        logger.info("Setting up RAG pipeline...")

        self.pipeline = self.factory.create_pipeline()
        self.evaluator = self.factory.create_evaluator()

        # Index documents (this will use cache if available)
        self.pipeline.index_documents(self.dataset.documents)

        logger.info("Pipeline setup complete")

    def run_evaluation(self, num_qa: int = None, save_results: bool = True) -> Dict[str, Any]:
        """Run the complete evaluation process"""
        if not self.pipeline:
            self.setup_pipeline()

        # Get num_qa from config if not provided
        if num_qa is None:
            dataset_config = self.factory.config.get_dataset_config()
            num_qa = dataset_config.get('evaluation_subset_size', 100)

        # Set random seed for reproducible results
        import random
        self.current_seed = 42
        random.seed(self.current_seed)

        # Get random subset of QA pairs
        if num_qa >= len(self.dataset.qa_pairs):
            qa_subset = self.dataset.qa_pairs
            logger.info(f"Using all {len(qa_subset)} QA pairs")
        else:
            qa_indices = random.sample(range(len(self.dataset.qa_pairs)), num_qa)
            qa_subset = [self.dataset.qa_pairs[i] for i in qa_indices]
            logger.info(
                f"Using random subset with seed={self.current_seed}: {len(qa_subset)} QA pairs from {len(self.dataset.qa_pairs)} total")

        logger.info(f"Starting evaluation with {len(qa_subset)} QA pairs...")

        start_time = time.time()
        results = []

        for i, qa_pair in enumerate(qa_subset, 1):
            question = qa_pair['question']
            expected_answer = qa_pair['ground_truth']

            logger.info(f"Evaluating QA pair {i}/{len(qa_subset)}: {question[:50]}...")

            # Query the pipeline
            query_result = self.pipeline.query(question)

            # Evaluate the result
            evaluation_result = self.evaluator.evaluate(
                question=question,
                expected_answer=expected_answer,
                actual_answer=query_result.response,
                context_chunks=query_result.chunks,
                query_time=query_result.metadata.get('query_time', 0.0)
            )

            # Add metadata
            evaluation_result.update({
                'question': question,
                'expected_answer': expected_answer,
                'actual_answer': query_result.response,
                **query_result.metadata,  # ← NEU: Pipeline-Metadaten übernehmen
                'pipeline_info': self.pipeline.get_pipeline_info()
            })

            results.append(evaluation_result)

        total_time = time.time() - start_time
        logger.info(f"Evaluation complete. Total time: {total_time:.2f}s")

        # Calculate summary statistics
        summary = self._calculate_summary(results, total_time)

        # Save results if requested
        if save_results:
            self._save_results(results, summary)

        return summary

    def _calculate_summary(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        if not results:
            return {}

        # Extract metrics
        precisions = [r.get('precision', 0) for r in results]
        recalls = [r.get('recall', 0) for r in results]
        f1_scores = [r.get('f1', 0) for r in results]
        query_times = [r.get('query_time', 0) for r in results]
        response_lengths = [len(r.get('actual_answer', '')) for r in results]

        # Calculate averages
        summary = {
            'avg_precision': sum(precisions) / len(precisions),
            'min_precision': min(precisions),
            'max_precision': max(precisions),
            'avg_recall': sum(recalls) / len(recalls),
            'min_recall': min(recalls),
            'max_recall': max(recalls),
            'avg_f1': sum(f1_scores) / len(f1_scores),
            'min_f1': min(f1_scores),
            'max_f1': max(f1_scores),
            'avg_query_time': sum(query_times) / len(query_times),
            'min_query_time': min(query_times),
            'max_query_time': max(query_times),
            'avg_response_length': sum(response_lengths) / len(response_lengths),
            'min_response_length': min(response_lengths),
            'max_response_length': max(response_lengths)
        }

        # Add RAGAS metrics if available
        if 'faithfulness' in results[0]:
            faithfulness_scores = [r.get('faithfulness', 0) for r in results]
            answer_relevance_scores = [r.get('answer_relevance', 0) for r in results]
            context_relevance_scores = [r.get('context_relevance', 0) for r in results]
            dsgvo_scores = [r.get('dsgvo_score', 0) for r in results]

            summary.update({
                'avg_faithfulness': sum(faithfulness_scores) / len(faithfulness_scores),
                'min_faithfulness': min(faithfulness_scores),
                'max_faithfulness': max(faithfulness_scores),
                'avg_answer_relevance': sum(answer_relevance_scores) / len(answer_relevance_scores),
                'min_answer_relevance': min(answer_relevance_scores),
                'max_answer_relevance': max(answer_relevance_scores),
                'avg_context_relevance': sum(context_relevance_scores) / len(context_relevance_scores),
                'min_context_relevance': min(context_relevance_scores),
                'max_context_relevance': max(context_relevance_scores),
                'avg_dsgvo_score': sum(dsgvo_scores) / len(dsgvo_scores),
                'min_dsgvo_score': min(dsgvo_scores),
                'max_dsgvo_score': max(dsgvo_scores)
            })

        # NEU: Context-Optimization Metadaten
        context_stats = {
            'truncation_rate': sum(1 for r in results if r.get('truncated', False)) / len(
                results) if results else 0,
            'avg_context_utilization': sum(r.get('context_utilization', 0) for r in results) / len(
                results) if results else 0,
            'avg_chunks_used': sum(r.get('chunks_used', 0) for r in results) / len(results) if results else 0,
            'total_chunks_wasted': sum(max(0, r.get('chunks_total', 0) - r.get('chunks_used', 0)) for r in results)
        }

        # Add pipeline and evaluation info
        summary.update({
            'pipeline_info': self.pipeline.get_pipeline_info(),
            'context_optimization': context_stats,
            'evaluation_info': {
                'enabled_evaluators': [e.__class__.__name__ for e in self.evaluator.evaluators],
                'total_metrics': len(summary) - 3  # Exclude pipeline_info and evaluation_info
            },
            'total_evaluation_time': total_time,
            'qa_pairs_evaluated': len(results),
            'random_seed': self.current_seed  # Verwendet die Instance Variable
        })

        return summary

    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Save evaluation results and summary to files with experiment name"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_filename = f"{self.experiment_name}_evaluation_results_{timestamp}.json"
        results_path = Path(results_filename)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_filename = f"{self.experiment_name}_evaluation_summary_{timestamp}.json"
        summary_path = Path(summary_filename)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Also save to cache
        self.factory.cache.save_evaluation_results(results, f"_{timestamp}")

        logger.info(f"Results saved to {results_filename} and {summary_filename}")
        logger.info(f"Results also cached for experiment: {self.experiment_name}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        return self.factory.get_cache_info()

    def clear_cache(self):
        """Clear the cache for this experiment"""
        self.factory.clear_cache()
