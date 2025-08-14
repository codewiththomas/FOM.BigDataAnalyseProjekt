from typing import List, Dict, Any
import time
import re
from .interfaces import EvaluationInterface, Chunk
import logging

logger = logging.getLogger(__name__)


class PrecisionRecallEvaluator(EvaluationInterface):
    """Traditional precision and recall evaluation"""

    def __init__(self, config: Dict[str, Any]):
        self.name = "precision-recall"
        logger.info("Initialized precision-recall evaluator")

    def evaluate(self, query: str, ground_truth: str, response: str,
                retrieved_chunks: List[Chunk]) -> Dict[str, float]:
        """Evaluate precision and recall"""
        # Simple keyword-based evaluation
        query_words = set(re.findall(r'\w+', query.lower()))
        gt_words = set(re.findall(r'\w+', ground_truth.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))

        # Calculate precision and recall
        relevant_words = query_words.intersection(gt_words)
        retrieved_relevant = response_words.intersection(relevant_words)

        precision = len(retrieved_relevant) / len(response_words) if response_words else 0.0
        recall = len(retrieved_relevant) / len(relevant_words) if relevant_words else 0.0

        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_metric_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': 'Traditional precision, recall, and F1 metrics',
            'metrics': ['precision', 'recall', 'f1']
        }


class TimingEvaluator(EvaluationInterface):
    """Performance timing evaluation"""

    def __init__(self, config: Dict[str, Any]):
        self.name = "timing"
        logger.info("Initialized timing evaluator")

    def evaluate(self, query: str, ground_truth: str, response: str,
                retrieved_chunks: List[Chunk]) -> Dict[str, float]:
        """Evaluate timing metrics"""
        # This would typically be measured during the actual query process
        # For now, we'll return placeholder values
        return {
            'query_time': 0.5,  # Placeholder
            'tokens_per_second': 100.0,  # Placeholder
            'response_length': len(response)
        }

    def get_metric_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': 'Performance timing metrics',
            'metrics': ['query_time', 'tokens_per_second', 'response_length']
        }


class RAGASEvaluator(EvaluationInterface):
    """RAGAS-style evaluation metrics"""

    def __init__(self, config: Dict[str, Any]):
        self.name = "ragas"
        logger.info("Initialized RAGAS evaluator")

    def evaluate(self, query: str, ground_truth: str, response: str,
                retrieved_chunks: List[Chunk]) -> Dict[str, float]:
        """Evaluate using RAGAS-style metrics"""
        # Placeholder implementation - in practice you'd use the actual RAGAS library
        # For now, we'll implement simplified versions

        # Faithfulness: how much the response is based on retrieved chunks
        faithfulness = self._calculate_faithfulness(response, retrieved_chunks)

        # Answer relevance: how relevant the response is to the query
        answer_relevance = self._calculate_answer_relevance(query, response)

        # Context relevance: how relevant retrieved chunks are to the query
        context_relevance = self._calculate_context_relevance(query, retrieved_chunks)

        return {
            'faithfulness': faithfulness,
            'answer_relevance': answer_relevance,
            'context_relevance': context_relevance
        }

    def _calculate_faithfulness(self, response: str, chunks: List[Chunk]) -> float:
        """Calculate how faithful the response is to retrieved chunks"""
        if not chunks:
            return 0.0

        # Simple word overlap calculation
        response_words = set(re.findall(r'\w+', response.lower()))
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(re.findall(r'\w+', chunk.text.lower()))

        if not response_words:
            return 0.0

        overlap = len(response_words.intersection(chunk_words))
        return min(1.0, overlap / len(response_words))

    def _calculate_answer_relevance(self, query: str, response: str) -> float:
        """Calculate how relevant the response is to the query"""
        query_words = set(re.findall(r'\w+', query.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))

        if not query_words or not response_words:
            return 0.0

        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words))

    def _calculate_context_relevance(self, query: str, chunks: List[Chunk]) -> float:
        """Calculate how relevant retrieved chunks are to the query"""
        if not chunks:
            return 0.0

        query_words = set(re.findall(r'\w+', query.lower()))
        total_relevance = 0.0

        for chunk in chunks:
            chunk_words = set(re.findall(r'\w+', chunk.text.lower()))
            if chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                relevance = min(1.0, overlap / len(query_words))
                total_relevance += relevance

        return total_relevance / len(chunks)

    def get_metric_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': 'RAGAS-style evaluation metrics',
            'metrics': ['faithfulness', 'answer_relevance', 'context_relevance']
        }


class EvaluationManager:
    """Manages multiple evaluation metrics"""

    def __init__(self, config: Dict[str, Any]):
        self.evaluators = []
        self._setup_evaluators(config)

    def _setup_evaluators(self, config: Dict[str, Any]):
        """Setup evaluators based on configuration"""
        evaluation_config = config.get('evaluation', {})
        enabled_metrics = evaluation_config.get('enabled_metrics', ['precision-recall', 'timing', 'ragas'])

        for metric in enabled_metrics:
            if metric == 'precision-recall':
                self.evaluators.append(PrecisionRecallEvaluator({}))
            elif metric == 'timing':
                self.evaluators.append(TimingEvaluator({}))
            elif metric == 'ragas':
                self.evaluators.append(RAGASEvaluator({}))
            else:
                logger.warning(f"Unknown evaluation metric: {metric}")

    def evaluate_query(self, query: str, ground_truth: str, response: str,
                      retrieved_chunks: List[Chunk]) -> Dict[str, float]:
        """Evaluate a single query using all enabled metrics"""
        results = {}

        for evaluator in self.evaluators:
            try:
                metric_results = evaluator.evaluate(query, ground_truth, response, retrieved_chunks)
                results.update(metric_results)
            except Exception as e:
                logger.error(f"Error in {evaluator.name} evaluator: {e}")
                # Add error indicators
                for metric in evaluator.get_metric_info()['metrics']:
                    results[f"{evaluator.name}_{metric}_error"] = -1.0

        return results

    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get information about all enabled evaluators"""
        return {
            'enabled_evaluators': [eval.name for eval in self.evaluators],
            'total_metrics': sum(len(eval.get_metric_info()['metrics']) for eval in self.evaluators)
        }
