from typing import List, Dict, Any, Optional
import numpy as np
import re
from collections import Counter
from .base_evaluator import BaseEvaluator


class GenerationEvaluator(BaseEvaluator):
    """
    Evaluator für Text-Generation-Performance.

    Berechnet verschiedene Generation-Metriken wie ROUGE-L, BLEU, Semantic Similarity
    und Exact Match.
    """

    def __init__(self, name: str = "generation_evaluator", **kwargs):
        """
        Initialisiert den Generation-Evaluator.

        Args:
            name: Name des Evaluators
            **kwargs: Zusätzliche Parameter
        """
        super().__init__(name, **kwargs)

    def evaluate(self, predictions: List[str], ground_truth: List[str],
                **kwargs) -> Dict[str, Any]:
        """
        Führt die Generation-Evaluierung durch.

        Args:
            predictions: Liste der generierten Texte
            ground_truth: Liste der Referenz-Texte
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit Generation-Metriken
        """
        if not self.validate_inputs(predictions, ground_truth):
            raise ValueError("Ungültige Eingabedaten für Generation-Evaluierung")

        results = {}

        # ROUGE-L
        rouge_l_scores = []
        for pred, gt in zip(predictions, ground_truth):
            rouge_l = self._calculate_rouge_l(pred, gt)
            rouge_l_scores.append(rouge_l)
        results["rouge_l"] = np.mean(rouge_l_scores)

        # BLEU Score
        bleu_scores = []
        for pred, gt in zip(predictions, ground_truth):
            bleu = self._calculate_bleu(pred, gt)
            bleu_scores.append(bleu)
        results["bleu"] = np.mean(bleu_scores)

        # Exact Match
        exact_matches = []
        for pred, gt in zip(predictions, ground_truth):
            exact_match = self._calculate_exact_match(pred, gt)
            exact_matches.append(exact_match)
        results["exact_match"] = np.mean(exact_matches)

        # Semantic Similarity (vereinfacht über Wort-Überlappung)
        semantic_similarities = []
        for pred, gt in zip(predictions, ground_truth):
            semantic_sim = self._calculate_semantic_similarity(pred, gt)
            semantic_similarities.append(semantic_sim)
        results["semantic_similarity"] = np.mean(semantic_similarities)

        # Durchschnittliche Textlängen
        results["avg_prediction_length"] = np.mean([len(pred.split()) for pred in predictions])
        results["avg_ground_truth_length"] = np.mean([len(gt.split()) for gt in ground_truth])

        # Längen-Verhältnis
        length_ratios = []
        for pred, gt in zip(predictions, ground_truth):
            pred_len = len(pred.split())
            gt_len = len(gt.split())
            if gt_len > 0:
                length_ratios.append(pred_len / gt_len)
        results["length_ratio"] = np.mean(length_ratios) if length_ratios else 0.0

        # Zusätzliche Metriken
        results["num_samples"] = len(predictions)

        return results

    def _calculate_rouge_l(self, prediction: str, ground_truth: str) -> float:
        """
        Berechnet ROUGE-L Score.

        Args:
            prediction: Generierter Text
            ground_truth: Referenz-Text

        Returns:
            ROUGE-L Score
        """
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(ground_truth)

        if not pred_tokens or not gt_tokens:
            return 0.0

        # Longest Common Subsequence (LCS)
        lcs_length = self._lcs_length(pred_tokens, gt_tokens)

        # ROUGE-L berechnen
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(gt_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _calculate_bleu(self, prediction: str, ground_truth: str, n: int = 4) -> float:
        """
        Berechnet BLEU Score.

        Args:
            prediction: Generierter Text
            ground_truth: Referenz-Text
            n: Maximale n-gram Größe

        Returns:
            BLEU Score
        """
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(ground_truth)

        if not pred_tokens or not gt_tokens:
            return 0.0

        # Brevity Penalty
        bp = self._brevity_penalty(pred_tokens, gt_tokens)

        # n-gram Präzision
        precisions = []
        for i in range(1, n + 1):
            precision = self._ngram_precision(pred_tokens, gt_tokens, i)
            precisions.append(precision)

        if any(p == 0 for p in precisions):
            return 0.0

        # Geometrisches Mittel der Präzisionen
        log_precisions = [np.log(p) for p in precisions]
        geo_mean = np.exp(np.mean(log_precisions))

        return bp * geo_mean

    def _calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Berechnet Exact Match.

        Args:
            prediction: Generierter Text
            ground_truth: Referenz-Text

        Returns:
            1.0 wenn exakt gleich, 0.0 sonst
        """
        pred_normalized = self._normalize_text(prediction)
        gt_normalized = self._normalize_text(ground_truth)

        return 1.0 if pred_normalized == gt_normalized else 0.0

    def _calculate_semantic_similarity(self, prediction: str, ground_truth: str) -> float:
        """
        Berechnet semantische Ähnlichkeit (vereinfacht über Wort-Überlappung).

        Args:
            prediction: Generierter Text
            ground_truth: Referenz-Text

        Returns:
            Semantic Similarity Score
        """
        pred_tokens = set(self._tokenize(prediction.lower()))
        gt_tokens = set(self._tokenize(ground_truth.lower()))

        if not pred_tokens or not gt_tokens:
            return 0.0

        # Jaccard Similarity
        intersection = len(pred_tokens & gt_tokens)
        union = len(pred_tokens | gt_tokens)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenisiert einen Text.

        Args:
            text: Zu tokenisierender Text

        Returns:
            Liste von Tokens
        """
        # Einfache Tokenisierung
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def _normalize_text(self, text: str) -> str:
        """
        Normalisiert einen Text für Vergleiche.

        Args:
            text: Zu normalisierender Text

        Returns:
            Normalisierter Text
        """
        # Kleinbuchstaben, Entfernung von Satzzeichen und extra Leerzeichen
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Berechnet die Länge der längsten gemeinsamen Subsequenz.

        Args:
            seq1: Erste Sequenz
            seq2: Zweite Sequenz

        Returns:
            Länge der LCS
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _brevity_penalty(self, pred_tokens: List[str], gt_tokens: List[str]) -> float:
        """
        Berechnet Brevity Penalty für BLEU.

        Args:
            pred_tokens: Tokens der Vorhersage
            gt_tokens: Tokens der Referenz

        Returns:
            Brevity Penalty
        """
        pred_len = len(pred_tokens)
        gt_len = len(gt_tokens)

        if pred_len >= gt_len:
            return 1.0
        else:
            return np.exp(1 - gt_len / pred_len)

    def _ngram_precision(self, pred_tokens: List[str], gt_tokens: List[str], n: int) -> float:
        """
        Berechnet n-gram Präzision.

        Args:
            pred_tokens: Tokens der Vorhersage
            gt_tokens: Tokens der Referenz
            n: n-gram Größe

        Returns:
            n-gram Präzision
        """
        if len(pred_tokens) < n:
            return 0.0

        pred_ngrams = self._get_ngrams(pred_tokens, n)
        gt_ngrams = self._get_ngrams(gt_tokens, n)

        if not pred_ngrams:
            return 0.0

        matches = 0
        for ngram in pred_ngrams:
            if ngram in gt_ngrams:
                matches += min(pred_ngrams[ngram], gt_ngrams[ngram])

        return matches / sum(pred_ngrams.values())

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """
        Erstellt n-grams aus Tokens.

        Args:
            tokens: Liste von Tokens
            n: n-gram Größe

        Returns:
            Counter mit n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)

        return Counter(ngrams)

    def get_metric_names(self) -> List[str]:
        """
        Gibt die Namen der berechneten Metriken zurück.

        Returns:
            Liste der Metrik-Namen
        """
        return [
            "rouge_l",
            "bleu",
            "exact_match",
            "semantic_similarity",
            "avg_prediction_length",
            "avg_ground_truth_length",
            "length_ratio",
            "num_samples"
        ]

    def evaluate_single_sample(self, prediction: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluiert ein einzelnes Beispiel.

        Args:
            prediction: Generierter Text
            ground_truth: Referenz-Text

        Returns:
            Dictionary mit Metriken für dieses Beispiel
        """
        return self.evaluate([prediction], [ground_truth])