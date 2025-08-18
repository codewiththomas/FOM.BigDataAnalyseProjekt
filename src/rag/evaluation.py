from typing import List, Dict, Any
import time
import re
from interfaces import EvaluationInterface, Chunk
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class PrecisionRecallEvaluator(EvaluationInterface):
    """Evaluator for precision, recall, and F1 score using improved metrics"""

    def __init__(self):
        logger.info("PrecisionRecallEvaluator initialized")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Evaluate precision, recall, and F1 score using multiple approaches"""
        if not expected_answer or not actual_answer:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Method 1: Word overlap (traditional)
        word_metrics = self._word_overlap_metrics(expected_answer, actual_answer)

        # Method 2: Sequence similarity
        sequence_similarity = self._sequence_similarity(expected_answer, actual_answer)

        # Method 3: Semantic similarity (using word embeddings approximation)
        semantic_similarity = self._semantic_similarity(expected_answer, actual_answer)

        # Combine metrics (weighted average)
        combined_precision = (0.4 * word_metrics['precision'] +
                            0.3 * sequence_similarity +
                            0.3 * semantic_similarity)

        combined_recall = (0.4 * word_metrics['recall'] +
                          0.3 * sequence_similarity +
                          0.3 * semantic_similarity)

        # Calculate F1
        combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0.0

        return {
            'precision': combined_precision,
            'recall': combined_recall,
            'f1': combined_f1,
            'word_precision': word_metrics['precision'],
            'word_recall': word_metrics['recall'],
            'sequence_similarity': sequence_similarity,
            'semantic_similarity': semantic_similarity
        }

    def _word_overlap_metrics(self, expected: str, actual: str) -> Dict[str, float]:
        """Calculate word-based precision and recall"""
        expected_words = set(re.findall(r'\w+', expected.lower()))
        actual_words = set(re.findall(r'\w+', actual.lower()))

        if not expected_words:
            return {'precision': 0.0, 'recall': 0.0}

        intersection = expected_words.intersection(actual_words)

        precision = len(intersection) / len(actual_words) if actual_words else 0.0
        recall = len(intersection) / len(expected_words)

        return {'precision': precision, 'recall': recall}

    def _sequence_similarity(self, expected: str, actual: str) -> float:
        """Calculate sequence similarity using difflib"""
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()

    def _semantic_similarity(self, expected: str, actual: str) -> float:
        """Calculate semantic similarity using word overlap with stemming approximation"""
        # Simple stemming approximation
        expected_stemmed = self._simple_stem(expected.lower())
        actual_stemmed = self._simple_stem(actual.lower())

        expected_words = set(re.findall(r'\w+', expected_stemmed))
        actual_words = set(re.findall(r'\w+', actual_stemmed))

        if not expected_words or not actual_words:
            return 0.0

        intersection = expected_words.intersection(actual_words)
        union = expected_words.union(actual_words)

        return len(intersection) / len(union) if union else 0.0

    def _simple_stem(self, text: str) -> str:
        """Simple stemming approximation"""
        # Remove common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es']
        words = text.split()
        stemmed_words = []

        for word in words:
            if len(word) > 3:
                for suffix in suffixes:
                    if word.endswith(suffix):
                        word = word[:-len(suffix)]
                        break
            stemmed_words.append(word)

        return ' '.join(stemmed_words)

    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information"""
        return {
            'name': 'precision-recall',
            'description': 'Improved precision, recall, and F1 metrics using multiple approaches',
            'metrics': ['precision', 'recall', 'f1', 'word_precision', 'word_recall', 'sequence_similarity', 'semantic_similarity']
        }


class TimingEvaluator(EvaluationInterface):
    """Evaluator for timing and performance metrics"""

    def __init__(self):
        logger.info("TimingEvaluator initialized")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Evaluate timing and performance metrics"""
        # Calculate tokens per second (improved estimation)
        estimated_tokens = self._estimate_tokens(actual_answer)
        tokens_per_second = estimated_tokens / query_time if query_time > 0 else 0.0

        # Calculate response quality metrics
        response_length = len(actual_answer)
        word_count = len(actual_answer.split())
        avg_word_length = sum(len(word) for word in actual_answer.split()) / word_count if word_count > 0 else 0

        return {
            'query_time': query_time,
            'tokens_per_second': tokens_per_second,
            'response_length': response_length,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'estimated_tokens': estimated_tokens
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count more accurately"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is a simplified approach - in production you'd use a proper tokenizer
        return len(text) // 4

    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information"""
        return {
            'name': 'timing',
            'description': 'Performance timing and response quality metrics',
            'metrics': ['query_time', 'tokens_per_second', 'response_length', 'word_count', 'avg_word_length', 'estimated_tokens']
        }


class RAGASEvaluator(EvaluationInterface):
    """Improved RAGAS-style evaluator"""

    def __init__(self):
        logger.info("RAGASEvaluator initialized")

    def evaluate(self, question: str, expected_answer: str, actual_answer: str,
                context_chunks: List[Chunk], query_time: float) -> Dict[str, Any]:
        """Evaluate RAGAS-style metrics with improved algorithms"""
        if not context_chunks:
            return {
                'faithfulness': 0.0,
                'answer_relevance': 0.0,
                'context_relevance': 0.0,
                'context_utilization': 0.0,
                'dsgvo_score': 0.0
            }

        # Faithfulness: How much the answer relies on the context
        faithfulness = self._calculate_faithfulness(actual_answer, context_chunks)

        # Answer Relevance: How relevant the answer is to the question
        answer_relevance = self._calculate_answer_relevance(question, actual_answer)

        # Context Relevance: How relevant retrieved chunks are to the question
        context_relevance = self._calculate_context_relevance(question, context_chunks)

        # Context Utilization: How well the context is used
        context_utilization = self._calculate_context_utilization(actual_answer, context_chunks)

        # DSGVO Score: Weighted combination of key metrics
        dsgvo_score = (faithfulness * 0.5 +
                       answer_relevance * 0.3 +
                       context_relevance * 0.2)

        return {
            'faithfulness': faithfulness,
            'answer_relevance': answer_relevance,
            'context_relevance': context_relevance,
            'context_utilization': context_utilization,
            'dsgvo_score': dsgvo_score
        }

    def _calculate_faithfulness(self, answer: str, chunks: List[Chunk]) -> float:
        """Calculate how faithful the answer is to the provided context"""
        if not answer or not chunks:
            return 0.0

        context_text = " ".join([chunk.text for chunk in chunks])
        context_words = set(re.findall(r'\w+', context_text.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))

        if not answer_words:
            return 0.0

        # Calculate word overlap
        context_overlap = len(answer_words.intersection(context_words))
        faithfulness_score = context_overlap / len(answer_words)

        # Boost score if answer is concise and relevant
        if len(answer) < 200:  # Short answers are often more faithful
            faithfulness_score *= 1.2

        return min(faithfulness_score, 1.0)

    def _calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question"""
        if not question or not answer:
            return 0.0

        question_words = set(re.findall(r'\w+', question.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))

        if not question_words or not answer_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(question_words.intersection(answer_words))
        union = len(question_words.union(answer_words))

        relevance = intersection / union if union > 0 else 0.0

        # Boost score for longer, more detailed answers
        if len(answer) > 100:
            relevance *= 1.1

        return min(relevance, 1.0)

    def _calculate_context_relevance(self, question: str, chunks: List[Chunk]) -> float:
        """Calculate how relevant the retrieved context is to the question"""
        if not question or not chunks:
            return 0.0

        question_words = set(re.findall(r'\w+', question.lower()))
        total_relevance = 0.0

        for chunk in chunks:
            chunk_words = set(re.findall(r'\w+', chunk.text.lower()))
            if chunk_words and question_words:
                intersection = len(question_words.intersection(chunk_words))
                relevance = intersection / len(question_words)
                total_relevance += relevance

        return total_relevance / len(chunks) if chunks else 0.0

    def _calculate_context_utilization(self, answer: str, chunks: List[Chunk]) -> float:
        """Calculate how well the context is utilized in the answer"""
        if not answer or not chunks:
            return 0.0

        context_text = " ".join([chunk.text for chunk in chunks])
        context_words = set(re.findall(r'\w+', context_text.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))

        if not context_words or not answer_words:
            return 0.0

        # Calculate utilization ratio
        utilized_words = answer_words.intersection(context_words)
        utilization = len(utilized_words) / len(context_words)

        # Penalize if answer is too short compared to context
        if len(answer) < len(context_text) * 0.1:
            utilization *= 0.8

        return min(utilization, 1.0)

    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information"""
        return {
            'name': 'ragas',
            'description': 'Improved RAGAS-style evaluation metrics',
            'metrics': ['faithfulness', 'answer_relevance', 'context_relevance', 'context_utilization', 'dsgvo_score']
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
