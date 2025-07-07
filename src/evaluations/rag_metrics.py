from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json


class RAGMetrics:
    """
    Umfassende Evaluierungsklasse für RAG-Systeme mit verschiedenen Metriken.
    """

    def __init__(self):
        self.metrics = {}

    def calculate_precision_recall_f1(self, retrieved_docs: List[str],
                                    relevant_docs: List[str],
                                    query: str = "") -> Dict[str, float]:
        """
        Berechnet Precision, Recall und F1-Score für Retrieval-Ergebnisse.
        """
        # Konvertiere zu binären Labels (1 = relevant, 0 = nicht relevant)
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        # True Positives, False Positives, False Negatives
        tp = len(retrieved_set.intersection(relevant_set))
        fp = len(retrieved_set - relevant_set)
        fn = len(relevant_set - retrieved_set)

        # Metriken berechnen
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

    def calculate_ragas_metrics(self,
                               context: List[str],
                               question: str,
                               answer: str,
                               ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Berechnet RAGAS-ähnliche Metriken für RAG-Systeme.
        """
        metrics = {}

        # Context Relevance Score (vereinfacht)
        context_relevance = self._calculate_context_relevance(context, question)
        metrics["context_relevance"] = context_relevance

        # Answer Relevance Score (vereinfacht)
        answer_relevance = self._calculate_answer_relevance(answer, question)
        metrics["answer_relevance"] = answer_relevance

        # Faithfulness Score (falls Ground Truth verfügbar)
        if ground_truth:
            faithfulness = self._calculate_faithfulness(answer, ground_truth)
            metrics["faithfulness"] = faithfulness

        # Answer Correctness (falls Ground Truth verfügbar)
        if ground_truth:
            correctness = self._calculate_answer_correctness(answer, ground_truth)
            metrics["answer_correctness"] = correctness

        return metrics

    def _calculate_context_relevance(self, context: List[str], question: str) -> float:
        """
        Berechnet die Relevanz des Kontexts zur Frage (vereinfacht).
        """
        if not context:
            return 0.0

        # Einfache Wortüberlappung als Proxy für Relevanz
        question_words = set(question.lower().split())
        total_relevance = 0.0

        for ctx in context:
            context_words = set(ctx.lower().split())
            overlap = len(question_words.intersection(context_words))
            relevance = overlap / len(question_words) if question_words else 0.0
            total_relevance += relevance

        return total_relevance / len(context)

    def _calculate_answer_relevance(self, answer: str, question: str) -> float:
        """
        Berechnet die Relevanz der Antwort zur Frage (vereinfacht).
        """
        if not answer or not question:
            return 0.0

        # Einfache Wortüberlappung als Proxy für Relevanz
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        overlap = len(question_words.intersection(answer_words))
        relevance = overlap / len(question_words) if question_words else 0.0

        return min(relevance, 1.0)

    def _calculate_faithfulness(self, answer: str, ground_truth: str) -> float:
        """
        Berechnet die Treue der Antwort zum Ground Truth (vereinfacht).
        """
        if not answer or not ground_truth:
            return 0.0

        # Einfache Ähnlichkeit basierend auf Wortüberlappung
        answer_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())

        intersection = len(answer_words.intersection(gt_words))
        union = len(answer_words.union(gt_words))

        return intersection / union if union > 0 else 0.0

    def _calculate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Berechnet die Korrektheit der Antwort (vereinfacht).
        """
        return self._calculate_faithfulness(answer, ground_truth)

    def calculate_retrieval_metrics(self,
                                  query: str,
                                  retrieved_docs: List[str],
                                  relevant_docs: List[str],
                                  top_k: int = 5) -> Dict[str, Any]:
        """
        Berechnet umfassende Retrieval-Metriken.
        """
        metrics = {}

        # Precision@k, Recall@k, F1@k
        for k in [1, 3, 5, 10]:
            if k <= len(retrieved_docs):
                k_docs = retrieved_docs[:k]
                k_metrics = self.calculate_precision_recall_f1(k_docs, relevant_docs)

                metrics[f"precision@{k}"] = k_metrics["precision"]
                metrics[f"recall@{k}"] = k_metrics["recall"]
                metrics[f"f1@{k}"] = k_metrics["f1_score"]

        # Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
        metrics["mrr"] = mrr

        # Normalized Discounted Cumulative Gain (nDCG)
        ndcg = self._calculate_ndcg(retrieved_docs, relevant_docs)
        metrics["ndcg"] = ndcg

        return metrics

    def _calculate_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Berechnet Mean Reciprocal Rank.
        """
        relevant_set = set(relevant_docs)

        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Berechnet Normalized Discounted Cumulative Gain (vereinfacht).
        """
        if not retrieved_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        dcg = 0.0
        idcg = 0.0

        # DCG berechnen
        for i, doc in enumerate(retrieved_docs):
            relevance = 1.0 if doc in relevant_set else 0.0
            dcg += relevance / np.log2(i + 2)  # +2 weil log2(1) = 0

        # IDCG berechnen (ideale Reihenfolge)
        num_relevant = len(relevant_set)
        for i in range(min(num_relevant, len(retrieved_docs))):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_rag_system(self,
                           queries: List[str],
                           retrieved_docs: List[List[str]],
                           relevant_docs: List[List[str]],
                           answers: List[str],
                           ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Führt eine umfassende Evaluierung des RAG-Systems durch.
        """
        results = {
            "retrieval_metrics": {},
            "generation_metrics": {},
            "overall_metrics": {}
        }

        # Retrieval-Metriken über alle Queries
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_mrrs = []
        all_ndcgs = []

        for i, (query, retrieved, relevant) in enumerate(zip(queries, retrieved_docs, relevant_docs)):
            retrieval_metrics = self.calculate_retrieval_metrics(query, retrieved, relevant)

            all_precisions.append(retrieval_metrics["precision@5"])
            all_recalls.append(retrieval_metrics["recall@5"])
            all_f1s.append(retrieval_metrics["f1@5"])
            all_mrrs.append(retrieval_metrics["mrr"])
            all_ndcgs.append(retrieval_metrics["ndcg"])

        # Durchschnittliche Retrieval-Metriken
        results["retrieval_metrics"] = {
            "avg_precision@5": np.mean(all_precisions),
            "avg_recall@5": np.mean(all_recalls),
            "avg_f1@5": np.mean(all_f1s),
            "avg_mrr": np.mean(all_mrrs),
            "avg_ndcg": np.mean(all_ndcgs)
        }

        # Generation-Metriken (falls Ground Truth verfügbar)
        if ground_truths:
            all_faithfulness = []
            all_correctness = []

            for answer, gt in zip(answers, ground_truths):
                gen_metrics = self.calculate_ragas_metrics([], "", answer, gt)
                all_faithfulness.append(gen_metrics.get("faithfulness", 0.0))
                all_correctness.append(gen_metrics.get("answer_correctness", 0.0))

            results["generation_metrics"] = {
                "avg_faithfulness": np.mean(all_faithfulness),
                "avg_correctness": np.mean(all_correctness)
            }

        # Gesamtbewertung
        results["overall_metrics"] = {
            "num_queries": len(queries),
            "avg_retrieval_score": np.mean(all_f1s),
            "avg_generation_score": np.mean(all_correctness) if ground_truths else None
        }

        return results

    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """
        Speichert Metriken in einer JSON-Datei.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Lädt Metriken aus einer JSON-Datei.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)