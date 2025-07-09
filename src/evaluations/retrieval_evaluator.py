from typing import List, Dict, Any, Set, Optional
import numpy as np
import math
from .base_evaluator import BaseEvaluator


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator für Retrieval-Performance.

    Berechnet verschiedene Retrieval-Metriken wie Precision@k, Recall@k, F1@k, MRR und NDCG.
    """

    def __init__(self, name: str = "retrieval_evaluator", k_values: List[int] = None, **kwargs):
        """
        Initialisiert den Retrieval-Evaluator.

        Args:
            name: Name des Evaluators
            k_values: Liste der k-Werte für @k-Metriken (default: [1, 3, 5, 10])
            **kwargs: Zusätzliche Parameter
        """
        super().__init__(name, **kwargs)
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(self, predictions: List[List[str]], ground_truth: List[List[str]],
                **kwargs) -> Dict[str, Any]:
        """
        Führt die Retrieval-Evaluierung durch.

        Args:
            predictions: Liste von Listen mit retrieval-Ergebnissen (pro Query)
            ground_truth: Liste von Listen mit relevanten Dokumenten (pro Query)
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit Retrieval-Metriken
        """
        if not self.validate_inputs(predictions, ground_truth):
            raise ValueError("Ungültige Eingabedaten für Retrieval-Evaluierung")

        results = {}

        # Precision@k, Recall@k, F1@k für alle k-Werte
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []

            for pred, gt in zip(predictions, ground_truth):
                precision = self._calculate_precision_at_k(pred, gt, k)
                recall = self._calculate_recall_at_k(pred, gt, k)
                f1 = self._calculate_f1_at_k(precision, recall)

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

            results[f"precision@{k}"] = np.mean(precision_scores)
            results[f"recall@{k}"] = np.mean(recall_scores)
            results[f"f1@{k}"] = np.mean(f1_scores)

        # Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for pred, gt in zip(predictions, ground_truth):
            mrr = self._calculate_mrr(pred, gt)
            mrr_scores.append(mrr)
        results["mrr"] = np.mean(mrr_scores)

        # NDCG für alle k-Werte
        for k in self.k_values:
            ndcg_scores = []
            for pred, gt in zip(predictions, ground_truth):
                ndcg = self._calculate_ndcg_at_k(pred, gt, k)
                ndcg_scores.append(ndcg)
            results[f"ndcg@{k}"] = np.mean(ndcg_scores)

        # Zusätzliche Metriken
        results["num_queries"] = len(predictions)
        results["avg_retrieved_docs"] = np.mean([len(pred) for pred in predictions])
        results["avg_relevant_docs"] = np.mean([len(gt) for gt in ground_truth])

        return results

    def _calculate_precision_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Berechnet Precision@k.

        Args:
            predictions: Liste der retrieval-Ergebnisse
            ground_truth: Liste der relevanten Dokumente
            k: Anzahl der zu betrachtenden Top-Ergebnisse

        Returns:
            Precision@k Wert
        """
        if not predictions or k <= 0:
            return 0.0

        top_k_predictions = predictions[:k]
        relevant_set = set(ground_truth)

        relevant_retrieved = sum(1 for doc in top_k_predictions if doc in relevant_set)

        return relevant_retrieved / len(top_k_predictions)

    def _calculate_recall_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Berechnet Recall@k.

        Args:
            predictions: Liste der retrieval-Ergebnisse
            ground_truth: Liste der relevanten Dokumente
            k: Anzahl der zu betrachtenden Top-Ergebnisse

        Returns:
            Recall@k Wert
        """
        if not ground_truth or k <= 0:
            return 0.0

        top_k_predictions = predictions[:k]
        relevant_set = set(ground_truth)

        relevant_retrieved = sum(1 for doc in top_k_predictions if doc in relevant_set)

        return relevant_retrieved / len(relevant_set)

    def _calculate_f1_at_k(self, precision: float, recall: float) -> float:
        """
        Berechnet F1@k aus Precision und Recall.

        Args:
            precision: Precision-Wert
            recall: Recall-Wert

        Returns:
            F1@k Wert
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _calculate_mrr(self, predictions: List[str], ground_truth: List[str]) -> float:
        """
        Berechnet Mean Reciprocal Rank (MRR).

        Args:
            predictions: Liste der retrieval-Ergebnisse
            ground_truth: Liste der relevanten Dokumente

        Returns:
            MRR Wert
        """
        relevant_set = set(ground_truth)

        for i, doc in enumerate(predictions):
            if doc in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_ndcg_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Berechnet Normalized Discounted Cumulative Gain (NDCG@k).

        Args:
            predictions: Liste der retrieval-Ergebnisse
            ground_truth: Liste der relevanten Dokumente
            k: Anzahl der zu betrachtenden Top-Ergebnisse

        Returns:
            NDCG@k Wert
        """
        if not predictions or k <= 0:
            return 0.0

        top_k_predictions = predictions[:k]
        relevant_set = set(ground_truth)

        # DCG berechnen
        dcg = 0.0
        for i, doc in enumerate(top_k_predictions):
            if doc in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 weil log2(1) = 0

        # IDCG berechnen (ideal DCG)
        idcg = 0.0
        num_relevant = min(len(ground_truth), k)
        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def get_metric_names(self) -> List[str]:
        """
        Gibt die Namen der berechneten Metriken zurück.

        Returns:
            Liste der Metrik-Namen
        """
        metrics = []

        # @k Metriken
        for k in self.k_values:
            metrics.extend([f"precision@{k}", f"recall@{k}", f"f1@{k}", f"ndcg@{k}"])

        # Weitere Metriken
        metrics.extend(["mrr", "num_queries", "avg_retrieved_docs", "avg_relevant_docs"])

        return metrics

    def evaluate_single_query(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """
        Evaluiert eine einzelne Query.

        Args:
            predictions: Retrieval-Ergebnisse für eine Query
            ground_truth: Relevante Dokumente für eine Query

        Returns:
            Dictionary mit Metriken für diese Query
        """
        return self.evaluate([predictions], [ground_truth])

    def calculate_statistical_significance(self, results1: Dict[str, Any],
                                         results2: Dict[str, Any],
                                         alpha: float = 0.05) -> Dict[str, bool]:
        """
        Berechnet statistische Signifikanz zwischen zwei Ergebnissen.

        Args:
            results1: Erste Ergebnisse
            results2: Zweite Ergebnisse
            alpha: Signifikanz-Level

        Returns:
            Dictionary mit Signifikanz-Tests für jede Metrik
        """
        # Vereinfachte Implementierung - könnte mit scipy.stats erweitert werden
        significance = {}

        for metric in self.get_metric_names():
            if metric in results1 and metric in results2:
                # Vereinfachter Test basierend auf Differenz
                diff = abs(results1[metric] - results2[metric])
                # Heuristische Schwelle - sollte durch echte statistische Tests ersetzt werden
                significance[metric] = diff > 0.05

        return significance