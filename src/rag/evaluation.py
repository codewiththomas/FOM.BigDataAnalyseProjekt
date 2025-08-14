from typing import List, Dict, Any
import time
import re
from interfaces import EvaluationInterface, Chunk
import logging

logger = logging.getLogger(__name__)


class PrecisionRecallEvaluator(EvaluationInterface):
    """Evaluator for precision, recall, and F1 score"""

    def __init__(self):
        logger.info("PrecisionRecallEvaluator initialized")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Evaluate precision, recall, and F1 score"""
        # Simple word overlap evaluation
        expected_words = set(expected_answer.lower().split())
        actual_words = set(actual_answer.lower().split())

        if not expected_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Calculate intersection
        intersection = expected_words.intersection(actual_words)

        # Calculate metrics
        precision = len(intersection) / len(actual_words) if actual_words else 0.0
        recall = len(intersection) / len(expected_words)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information (required by interface)"""
        return {
            'name': 'precision-recall',
            'description': 'Traditional precision, recall, and F1 metrics',
            'metrics': ['precision', 'recall', 'f1']
        }


class TimingEvaluator(EvaluationInterface):
    """Evaluator for timing and performance metrics"""

    def __init__(self):
        logger.info("TimingEvaluator initialized")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Evaluate timing and performance metrics"""
        # Calculate tokens per second (placeholder)
        # In a real implementation, you'd count actual tokens
        estimated_tokens = len(actual_answer.split()) * 1.3  # Rough estimate
        tokens_per_second = estimated_tokens / query_time if query_time > 0 else 0.0

        return {
            'query_time': query_time,
            'tokens_per_second': tokens_per_second,
            'response_length': len(actual_answer)
        }

    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information (required by interface)"""
        return {
            'name': 'timing',
            'description': 'Performance timing metrics',
            'metrics': ['query_time', 'tokens_per_second', 'response_length']
        }


class RAGASEvaluator(EvaluationInterface):
    """Simplified RAGAS-style evaluator"""

    def __init__(self):
        logger.info("RAGASEvaluator initialized")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Evaluate RAGAS-style metrics"""
        # Faithfulness: How much the answer relies on the context
        context_text = " ".join([chunk.text for chunk in context_chunks])
        context_words = set(context_text.lower().split())
        answer_words = set(actual_answer.lower().split())

        if answer_words:
            context_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
            faithfulness = min(context_overlap * 2, 1.0)  # Scale to 0-1
        else:
            faithfulness = 0.0

        # Answer Relevance: How relevant the answer is to the question
        question_words = set(question.lower().split())
        if answer_words:
            question_answer_overlap = len(answer_words.intersection(question_words)) / len(answer_words)
            answer_relevance = min(question_answer_overlap * 3, 1.0)  # Scale to 0-1
        else:
            answer_relevance = 0.0

        # Context Relevance: How relevant the retrieved context is
        if context_words and question_words:
            context_question_overlap = len(context_words.intersection(question_words)) / len(context_words)
            context_relevance = min(context_question_overlap * 2, 1.0)  # Scale to 0-1
        else:
            context_relevance = 0.0

        return {
            'faithfulness': faithfulness,
            'answer_relevance': answer_relevance,
            'context_relevance': context_relevance
        }

    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information (required by interface)"""
        return {
            'name': 'ragas',
            'description': 'RAGAS-style evaluation metrics',
            'metrics': ['faithfulness', 'answer_relevance', 'context_relevance']
        }


class EvaluationManager:
    """Manages multiple evaluation metrics"""

    def __init__(self, evaluators: List[EvaluationInterface]):
        self.evaluators = evaluators
        logger.info(f"EvaluationManager initialized with {len(evaluators)} evaluators")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Run all enabled evaluators and combine results"""
        combined_results = {}

        for evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(
                    question=question,
                    expected_answer=expected_answer,
                    actual_answer=actual_answer,
                    context_chunks=context_chunks,
                    query_time=query_time
                )
                combined_results.update(result)
            except Exception as e:
                logger.error(f"Error in evaluator {evaluator.__class__.__name__}: {e}")
                # Continue with other evaluators

        return combined_results

    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get information about enabled evaluators"""
        return {
            'enabled_evaluators': [e.__class__.__name__ for e in self.evaluators],
            'total_evaluators': len(self.evaluators)
        }
