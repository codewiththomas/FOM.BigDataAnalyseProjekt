from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
import json
from pathlib import Path


class BaseEvaluator(ABC):
    """
    Abstrakte Basisklasse für alle Evaluierungskomponenten.

    Diese Klasse definiert das Interface für verschiedene Evaluierungstypen
    und stellt gemeinsame Funktionalitäten bereit.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialisiert den Evaluator.

        Args:
            name: Name des Evaluators
            **kwargs: Zusätzliche evaluator-spezifische Parameter
        """
        self.name = name
        self.config = kwargs
        self.results_history = []

    @abstractmethod
    def evaluate(self, predictions: List[Any], ground_truth: List[Any],
                **kwargs) -> Dict[str, Any]:
        """
        Führt die Evaluierung durch.

        Args:
            predictions: Vorhersagen/Ergebnisse des Systems
            ground_truth: Referenz-/Ground-Truth-Daten
            **kwargs: Zusätzliche Parameter für die Evaluierung

        Returns:
            Dictionary mit Evaluierungsergebnissen
        """
        pass

    def batch_evaluate(self, prediction_batches: List[List[Any]],
                      ground_truth_batches: List[List[Any]],
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Führt Evaluierung für mehrere Batches durch.

        Args:
            prediction_batches: Liste von Vorhersage-Batches
            ground_truth_batches: Liste von Ground-Truth-Batches
            **kwargs: Zusätzliche Parameter

        Returns:
            Liste von Evaluierungsergebnissen
        """
        results = []
        for pred_batch, gt_batch in zip(prediction_batches, ground_truth_batches):
            batch_result = self.evaluate(pred_batch, gt_batch, **kwargs)
            results.append(batch_result)
        return results

    def evaluate_with_metadata(self, predictions: List[Any], ground_truth: List[Any],
                             metadata: Optional[Dict[str, Any]] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Führt Evaluierung mit zusätzlichen Metadaten durch.

        Args:
            predictions: Vorhersagen/Ergebnisse des Systems
            ground_truth: Referenz-/Ground-Truth-Daten
            metadata: Zusätzliche Metadaten für die Evaluierung
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit Evaluierungsergebnissen und Metadaten
        """
        start_time = time.time()

        # Evaluierung durchführen
        results = self.evaluate(predictions, ground_truth, **kwargs)

        # Metadaten hinzufügen
        evaluation_metadata = {
            "evaluator_name": self.name,
            "evaluator_type": self.__class__.__name__,
            "evaluation_time": time.time() - start_time,
            "num_predictions": len(predictions),
            "num_ground_truth": len(ground_truth),
            "timestamp": time.time(),
            "config": self.config
        }

        if metadata:
            evaluation_metadata.update(metadata)

        results["metadata"] = evaluation_metadata

        # Zu Historie hinzufügen
        self.results_history.append(results)

        return results

    def get_metric_names(self) -> List[str]:
        """
        Gibt die Namen der berechneten Metriken zurück.

        Returns:
            Liste der Metrik-Namen
        """
        # Standardimplementierung - kann in Subklassen überschrieben werden
        return ["score"]

    def get_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration des Evaluators zurück.

        Returns:
            Dictionary mit Konfigurationsparametern
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            **self.config
        }

    def save_results(self, results: Dict[str, Any], file_path: str) -> None:
        """
        Speichert Evaluierungsergebnisse in eine Datei.

        Args:
            results: Evaluierungsergebnisse
            file_path: Pfad zur Ausgabedatei
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        Lädt Evaluierungsergebnisse aus einer Datei.

        Args:
            file_path: Pfad zur Eingabedatei

        Returns:
            Dictionary mit Evaluierungsergebnissen
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_results_history(self) -> List[Dict[str, Any]]:
        """
        Gibt die Historie aller Evaluierungsergebnisse zurück.

        Returns:
            Liste aller bisherigen Evaluierungsergebnisse
        """
        return self.results_history.copy()

    def clear_history(self) -> None:
        """
        Löscht die Ergebnishistorie.
        """
        self.results_history.clear()

    def validate_inputs(self, predictions: List[Any], ground_truth: List[Any]) -> bool:
        """
        Validiert die Eingabedaten für die Evaluierung.

        Args:
            predictions: Vorhersagen/Ergebnisse des Systems
            ground_truth: Referenz-/Ground-Truth-Daten

        Returns:
            True wenn die Eingaben valide sind
        """
        if not predictions or not ground_truth:
            return False

        if len(predictions) != len(ground_truth):
            return False

        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()