from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .base_evaluator import BaseEvaluator
from .retrieval_evaluator import RetrievalEvaluator
from .generation_evaluator import GenerationEvaluator
from .performance_evaluator import PerformanceEvaluator


class RAGEvaluator(BaseEvaluator):
    """
    End-to-End Evaluator für RAG-Systeme.

    Kombiniert Retrieval-, Generation- und Performance-Evaluierung und
    berechnet RAG-spezifische Metriken wie Faithfulness und Groundedness.
    """

    def __init__(self, name: str = "rag_evaluator", k_values: List[int] = None, **kwargs):
        """
        Initialisiert den RAG-Evaluator.

        Args:
            name: Name des Evaluators
            k_values: Liste der k-Werte für Retrieval-Metriken
            **kwargs: Zusätzliche Parameter
        """
        super().__init__(name, **kwargs)
        self.k_values = k_values or [1, 3, 5, 10]

        # Sub-Evaluatoren
        self.retrieval_evaluator = RetrievalEvaluator(k_values=self.k_values)
        self.generation_evaluator = GenerationEvaluator()
        self.performance_evaluator = PerformanceEvaluator()

    def evaluate(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]],
                **kwargs) -> Dict[str, Any]:
        """
        Führt die End-to-End RAG-Evaluierung durch.

        Args:
            predictions: Liste von RAG-Ergebnissen mit Struktur:
                {
                    "question": str,
                    "answer": str,
                    "retrieved_contexts": List[Dict],
                    "query_time": float,
                    "metadata": Dict
                }
            ground_truth: Liste von Ground-Truth-Daten mit Struktur:
                {
                    "question": str,
                    "gold_answer": str,
                    "relevant_chunks": List[str],
                    "category": str,
                    "difficulty": str
                }
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit kombinierten RAG-Metriken
        """
        if not self.validate_inputs(predictions, ground_truth):
            raise ValueError("Ungültige Eingabedaten für RAG-Evaluierung")

        results = {}

        # 1. Retrieval-Evaluierung
        retrieval_results = self._evaluate_retrieval(predictions, ground_truth)
        results.update({f"retrieval_{k}": v for k, v in retrieval_results.items()})

        # 2. Generation-Evaluierung
        generation_results = self._evaluate_generation(predictions, ground_truth)
        results.update({f"generation_{k}": v for k, v in generation_results.items()})

        # 3. Performance-Evaluierung
        performance_results = self._evaluate_performance(predictions)
        results.update({f"performance_{k}": v for k, v in performance_results.items()})

        # 4. RAG-spezifische Metriken
        rag_specific_results = self._evaluate_rag_specific(predictions, ground_truth)
        results.update(rag_specific_results)

        # 5. Kombinierte Metriken
        combined_results = self._calculate_combined_metrics(results)
        results.update(combined_results)

        # 6. Kategorien-spezifische Analyse (optional, nur wenn Kategorien vorhanden)
        if any("category" in gt for gt in ground_truth):
            category_results = self._analyze_by_category_simple(predictions, ground_truth)
            results["category_analysis"] = category_results

        return results

    def _evaluate_retrieval(self, predictions: List[Dict[str, Any]],
                           ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluiert die Retrieval-Performance.

        Args:
            predictions: RAG-Ergebnisse
            ground_truth: Ground-Truth-Daten

        Returns:
            Dictionary mit Retrieval-Metriken
        """
        # Retrieval-Daten extrahieren
        retrieved_docs = []
        relevant_docs = []

        for pred, gt in zip(predictions, ground_truth):
            # Extrahiere IDs der abgerufenen Dokumente
            if "retrieved_contexts" in pred:
                retrieved = [str(ctx.get("chunk_id", idx)) for idx, ctx in enumerate(pred["retrieved_contexts"])]
                retrieved_docs.append(retrieved)
            else:
                retrieved_docs.append([])

            # Extrahiere relevante Dokument-IDs
            if "relevant_chunks" in gt:
                relevant = [str(chunk_id) for chunk_id in gt["relevant_chunks"]]
                relevant_docs.append(relevant)
            else:
                relevant_docs.append([])

        return self.retrieval_evaluator.evaluate(retrieved_docs, relevant_docs)

    def _evaluate_generation(self, predictions: List[Dict[str, Any]],
                            ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluiert die Generation-Performance.

        Args:
            predictions: RAG-Ergebnisse
            ground_truth: Ground-Truth-Daten

        Returns:
            Dictionary mit Generation-Metriken
        """
        # Generation-Daten extrahieren
        generated_answers = [pred.get("answer", "") for pred in predictions]
        gold_answers = [gt.get("gold_answer", "") for gt in ground_truth]

        return self.generation_evaluator.evaluate(generated_answers, gold_answers)

    def _evaluate_performance(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluiert die Performance-Metriken.

        Args:
            predictions: RAG-Ergebnisse

        Returns:
            Dictionary mit Performance-Metriken
        """
        # Performance-Daten extrahieren
        latencies = [pred.get("query_time", 0.0) for pred in predictions]
        timestamps = [pred.get("timestamp", 0.0) for pred in predictions]

        return self.performance_evaluator.evaluate(
            predictions, predictions,
            latencies=latencies,
            timestamps=timestamps
        )

    def _evaluate_rag_specific(self, predictions: List[Dict[str, Any]],
                              ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluiert RAG-spezifische Metriken.

        Args:
            predictions: RAG-Ergebnisse
            ground_truth: Ground-Truth-Daten

        Returns:
            Dictionary mit RAG-spezifischen Metriken
        """
        results = {}

        # Faithfulness (Treue zur Quelle)
        faithfulness_scores = []
        for pred in predictions:
            faithfulness = self._calculate_faithfulness(pred)
            faithfulness_scores.append(faithfulness)
        results["faithfulness"] = np.mean(faithfulness_scores)

        # Groundedness (Verankerung in den Quellen)
        groundedness_scores = []
        for pred in predictions:
            groundedness = self._calculate_groundedness(pred)
            groundedness_scores.append(groundedness)
        results["groundedness"] = np.mean(groundedness_scores)

        # Answer Relevance (Antwort-Relevanz)
        relevance_scores = []
        for pred, gt in zip(predictions, ground_truth):
            relevance = self._calculate_answer_relevance(pred, gt)
            relevance_scores.append(relevance)
        results["answer_relevance"] = np.mean(relevance_scores)

        # Context Precision (Kontext-Präzision)
        context_precision_scores = []
        for pred, gt in zip(predictions, ground_truth):
            context_precision = self._calculate_context_precision(pred, gt)
            context_precision_scores.append(context_precision)
        results["context_precision"] = np.mean(context_precision_scores)

        # Context Recall (Kontext-Recall)
        context_recall_scores = []
        for pred, gt in zip(predictions, ground_truth):
            context_recall = self._calculate_context_recall(pred, gt)
            context_recall_scores.append(context_recall)
        results["context_recall"] = np.mean(context_recall_scores)

        return results

    def _calculate_faithfulness(self, prediction: Dict[str, Any]) -> float:
        """
        Berechnet Faithfulness (Treue zur Quelle).

        Args:
            prediction: RAG-Ergebnis

        Returns:
            Faithfulness Score
        """
        answer = prediction.get("answer", "")
        contexts = prediction.get("retrieved_contexts", [])

        if not answer or not contexts:
            return 0.0

        # Vereinfachte Faithfulness-Berechnung über Wort-Überlappung
        answer_words = set(answer.lower().split())
        context_words = set()

        for context in contexts:
            context_text = context.get("text", "")
            context_words.update(context_text.lower().split())

        if not context_words:
            return 0.0

        # Anteil der Antwort-Wörter, die in den Kontexten vorkommen
        overlap = len(answer_words & context_words)
        return overlap / len(answer_words) if answer_words else 0.0

    def _calculate_groundedness(self, prediction: Dict[str, Any]) -> float:
        """
        Berechnet Groundedness (Verankerung in den Quellen).

        Args:
            prediction: RAG-Ergebnis

        Returns:
            Groundedness Score
        """
        answer = prediction.get("answer", "")
        contexts = prediction.get("retrieved_contexts", [])

        if not answer or not contexts:
            return 0.0

        # Vereinfachte Groundedness-Berechnung
        # Prüft, ob die Antwort durch die Kontexte gestützt wird
        answer_sentences = answer.split('.')
        supported_sentences = 0

        for sentence in answer_sentences:
            if sentence.strip():
                # Prüfe, ob Satz durch Kontext gestützt wird
                sentence_words = set(sentence.lower().split())

                for context in contexts:
                    context_text = context.get("text", "")
                    context_words = set(context_text.lower().split())

                    # Wenn genügend Wörter übereinstimmen, ist der Satz gestützt
                    if len(sentence_words & context_words) / len(sentence_words) > 0.5:
                        supported_sentences += 1
                        break

        return supported_sentences / len(answer_sentences) if answer_sentences else 0.0

    def _calculate_answer_relevance(self, prediction: Dict[str, Any],
                                   ground_truth: Dict[str, Any]) -> float:
        """
        Berechnet Answer Relevance.

        Args:
            prediction: RAG-Ergebnis
            ground_truth: Ground-Truth-Daten

        Returns:
            Answer Relevance Score
        """
        question = prediction.get("question", "")
        answer = prediction.get("answer", "")

        if not question or not answer:
            return 0.0

        # Vereinfachte Relevanz-Berechnung über Wort-Überlappung
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Entferne Stopwörter (vereinfacht)
        stopwords = {"der", "die", "das", "und", "oder", "aber", "ist", "sind", "war", "waren", "hat", "haben"}
        question_words -= stopwords
        answer_words -= stopwords

        if not question_words:
            return 0.0

        overlap = len(question_words & answer_words)
        return overlap / len(question_words)

    def _calculate_context_precision(self, prediction: Dict[str, Any],
                                    ground_truth: Dict[str, Any]) -> float:
        """
        Berechnet Context Precision.

        Args:
            prediction: RAG-Ergebnis
            ground_truth: Ground-Truth-Daten

        Returns:
            Context Precision Score
        """
        contexts = prediction.get("retrieved_contexts", [])
        relevant_chunks = ground_truth.get("relevant_chunks", [])

        if not contexts:
            return 0.0

        relevant_set = set(str(chunk_id) for chunk_id in relevant_chunks)
        retrieved_relevant = 0

        for context in contexts:
            context_id = str(context.get("chunk_id", ""))
            if context_id in relevant_set:
                retrieved_relevant += 1

        return retrieved_relevant / len(contexts)

    def _calculate_context_recall(self, prediction: Dict[str, Any],
                                 ground_truth: Dict[str, Any]) -> float:
        """
        Berechnet Context Recall.

        Args:
            prediction: RAG-Ergebnis
            ground_truth: Ground-Truth-Daten

        Returns:
            Context Recall Score
        """
        contexts = prediction.get("retrieved_contexts", [])
        relevant_chunks = ground_truth.get("relevant_chunks", [])

        if not relevant_chunks:
            return 0.0

        relevant_set = set(str(chunk_id) for chunk_id in relevant_chunks)
        retrieved_set = set(str(ctx.get("chunk_id", "")) for ctx in contexts)

        retrieved_relevant = len(relevant_set & retrieved_set)
        return retrieved_relevant / len(relevant_set)

    def _calculate_combined_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Berechnet kombinierte Metriken.

        Args:
            results: Bisherige Evaluierungsergebnisse

        Returns:
            Dictionary mit kombinierten Metriken
        """
        combined = {}

        # RAG Score (gewichteter Durchschnitt)
        retrieval_score = results.get("retrieval_f1@5", 0.0)
        generation_score = results.get("generation_rouge_l", 0.0)
        faithfulness_score = results.get("faithfulness", 0.0)

        rag_score = (0.3 * retrieval_score + 0.3 * generation_score + 0.4 * faithfulness_score)
        combined["rag_score"] = rag_score

        # Quality Score (Qualitätsbewertung)
        groundedness_score = results.get("groundedness", 0.0)
        answer_relevance_score = results.get("answer_relevance", 0.0)

        quality_score = (0.5 * groundedness_score + 0.5 * answer_relevance_score)
        combined["quality_score"] = quality_score

        # Efficiency Score (Effizienz)
        avg_latency = results.get("performance_avg_latency", 1.0)
        throughput = results.get("performance_throughput_qps", 1.0)

        # Normalisierte Effizienz (niedrigere Latenz = besser)
        efficiency_score = min(1.0, throughput / max(avg_latency, 0.1))
        combined["efficiency_score"] = efficiency_score

        # Overall Score (Gesamtbewertung)
        overall_score = (0.4 * rag_score + 0.4 * quality_score + 0.2 * efficiency_score)
        combined["overall_score"] = overall_score

        return combined

    def _analyze_by_category_simple(self, predictions: List[Dict[str, Any]],
                                   ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Einfache Analyse nach Kategorien ohne Rekursion.

        Args:
            predictions: RAG-Ergebnisse
            ground_truth: Ground-Truth-Daten

        Returns:
            Dictionary mit kategorienspezifischen Analysen
        """
        category_results = {}

        # Gruppiere nach Kategorien
        categories = {}
        for pred, gt in zip(predictions, ground_truth):
            category = gt.get("category", "unknown")
            if category not in categories:
                categories[category] = {"predictions": [], "ground_truth": []}
            categories[category]["predictions"].append(pred)
            categories[category]["ground_truth"].append(gt)

        # Einfache Statistiken pro Kategorie
        for category, data in categories.items():
            if data["predictions"]:
                # Einfache Metriken ohne vollständige Evaluierung
                category_results[category] = {
                    "count": len(data["predictions"]),
                    "avg_query_time": sum(p.get("query_time", 0.0) for p in data["predictions"]) / len(data["predictions"]),
                    "avg_contexts_retrieved": sum(len(p.get("retrieved_contexts", [])) for p in data["predictions"]) / len(data["predictions"])
                }

        return category_results

    def get_metric_names(self) -> List[str]:
        """
        Gibt die Namen der berechneten Metriken zurück.

        Returns:
            Liste der Metrik-Namen
        """
        metrics = []

        # Retrieval-Metriken
        for k in self.k_values:
            metrics.extend([f"retrieval_precision@{k}", f"retrieval_recall@{k}", f"retrieval_f1@{k}"])

        # Generation-Metriken
        metrics.extend([
            "generation_rouge_l", "generation_bleu", "generation_exact_match",
            "generation_semantic_similarity"
        ])

        # Performance-Metriken
        metrics.extend([
            "performance_avg_latency", "performance_throughput_qps",
            "performance_avg_memory_mb"
        ])

        # RAG-spezifische Metriken
        metrics.extend([
            "faithfulness", "groundedness", "answer_relevance",
            "context_precision", "context_recall"
        ])

        # Kombinierte Metriken
        metrics.extend([
            "rag_score", "quality_score", "efficiency_score", "overall_score"
        ])

        return metrics