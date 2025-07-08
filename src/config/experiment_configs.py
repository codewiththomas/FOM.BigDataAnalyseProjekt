from typing import Dict, Any, Optional, List
from .base_config import BaseConfig


class ExperimentConfig(BaseConfig):
    """
    Konfiguration für Experimente.
    
    Definiert Parameter für Evaluierungen, Metriken und Experiment-Setups.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialisiert die Experiment-Konfiguration.
        
        Args:
            config_dict: Optionales Dictionary mit Konfigurationswerten
        """
        super().__init__(config_dict)
        self.merge_with_defaults()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Gibt die Standard-Experiment-Konfiguration zurück.
        
        Returns:
            Dictionary mit Standard-Konfigurationswerten
        """
        return {
            "experiment": {
                "name": "baseline_experiment",
                "version": "1.0.0",
                "description": "Baseline Experiment für RAG-System",
                "author": "FOM Research Team",
                "date": None,  # Wird automatisch gesetzt
                "seed": 42
            },
            "data": {
                "source_file": "data/raw/dsgvo.txt",
                "qa_file": "data/evaluation/qa_pairs.json",
                "processed_chunks_dir": "data/processed/chunks",
                "processed_embeddings_dir": "data/processed/embeddings"
            },
            "evaluation": {
                "metrics": {
                    "retrieval": ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg"],
                    "generation": ["rouge_l", "bleu", "bert_score", "semantic_similarity", "exact_match"],
                    "rag": ["faithfulness", "groundedness", "ragas_score"],
                    "performance": ["inference_time", "memory_usage", "throughput"]
                },
                "k_values": [1, 3, 5, 10],
                "batch_size": 32,
                "save_predictions": True,
                "save_retrieved_contexts": True
            },
            "output": {
                "results_dir": "data/evaluation/results",
                "save_detailed_results": True,
                "save_summary_only": False,
                "export_formats": ["json", "csv"],
                "include_visualizations": True
            },
            "logging": {
                "level": "INFO",
                "file": "logs/experiment.log",
                "console": True,
                "include_timestamps": True
            }
        }
    
    def _validate_config(self) -> None:
        """
        Validiert die Experiment-Konfiguration.
        """
        required_sections = ["experiment", "data", "evaluation", "output"]
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Erforderliche Sektion '{section}' fehlt in der Konfiguration")
        
        # Experiment-Validierung
        experiment_config = self._config["experiment"]
        if "name" not in experiment_config:
            raise ValueError("Experiment-Name ist erforderlich")
        
        if experiment_config.get("seed") is not None:
            if not isinstance(experiment_config["seed"], int):
                raise ValueError("Seed muss eine Ganzzahl sein")
        
        # Data-Validierung
        data_config = self._config["data"]
        if "source_file" not in data_config:
            raise ValueError("Quelldatei ist erforderlich")
        
        # Evaluation-Validierung
        evaluation_config = self._config["evaluation"]
        if "metrics" not in evaluation_config:
            raise ValueError("Metriken sind erforderlich")
        
        metrics = evaluation_config["metrics"]
        valid_retrieval_metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg"]
        valid_generation_metrics = ["rouge_l", "bleu", "bert_score", "semantic_similarity", "exact_match"]
        valid_rag_metrics = ["faithfulness", "groundedness", "ragas_score"]
        valid_performance_metrics = ["inference_time", "memory_usage", "throughput"]
        
        for metric in metrics.get("retrieval", []):
            if metric not in valid_retrieval_metrics:
                raise ValueError(f"Ungültige Retrieval-Metrik: {metric}")
        
        for metric in metrics.get("generation", []):
            if metric not in valid_generation_metrics:
                raise ValueError(f"Ungültige Generation-Metrik: {metric}")
        
        for metric in metrics.get("rag", []):
            if metric not in valid_rag_metrics:
                raise ValueError(f"Ungültige RAG-Metrik: {metric}")
        
        for metric in metrics.get("performance", []):
            if metric not in valid_performance_metrics:
                raise ValueError(f"Ungültige Performance-Metrik: {metric}")
        
        # K-Werte validieren
        k_values = evaluation_config.get("k_values", [])
        if not all(isinstance(k, int) and k > 0 for k in k_values):
            raise ValueError("K-Werte müssen positive Ganzzahlen sein")
        
        # Batch-Größe validieren
        batch_size = evaluation_config.get("batch_size", 1)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch-Größe muss eine positive Ganzzahl sein")
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Gibt Experiment-Informationen zurück.
        
        Returns:
            Dictionary mit Experiment-Informationen
        """
        return self._config["experiment"].copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Gibt die Daten-Konfiguration zurück.
        
        Returns:
            Dictionary mit Daten-Konfiguration
        """
        return self._config["data"].copy()
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Gibt die Evaluierungs-Konfiguration zurück.
        
        Returns:
            Dictionary mit Evaluierungs-Konfiguration
        """
        return self._config["evaluation"].copy()
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Gibt die Output-Konfiguration zurück.
        
        Returns:
            Dictionary mit Output-Konfiguration
        """
        return self._config["output"].copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Gibt die Logging-Konfiguration zurück.
        
        Returns:
            Dictionary mit Logging-Konfiguration
        """
        return self._config.get("logging", {}).copy()
    
    def get_enabled_metrics(self) -> Dict[str, List[str]]:
        """
        Gibt alle aktivierten Metriken zurück.
        
        Returns:
            Dictionary mit aktivierten Metriken nach Kategorie
        """
        return self._config["evaluation"]["metrics"].copy()
    
    def enable_metric(self, category: str, metric: str) -> None:
        """
        Aktiviert eine Metrik.
        
        Args:
            category: Kategorie der Metrik ("retrieval", "generation", "rag", "performance")
            metric: Name der Metrik
        """
        if category not in self._config["evaluation"]["metrics"]:
            self._config["evaluation"]["metrics"][category] = []
        
        if metric not in self._config["evaluation"]["metrics"][category]:
            self._config["evaluation"]["metrics"][category].append(metric)
        
        self._validate_config()
    
    def disable_metric(self, category: str, metric: str) -> None:
        """
        Deaktiviert eine Metrik.
        
        Args:
            category: Kategorie der Metrik
            metric: Name der Metrik
        """
        if category in self._config["evaluation"]["metrics"]:
            if metric in self._config["evaluation"]["metrics"][category]:
                self._config["evaluation"]["metrics"][category].remove(metric)
    
    def set_k_values(self, k_values: List[int]) -> None:
        """
        Setzt die K-Werte für Retrieval-Metriken.
        
        Args:
            k_values: Liste von K-Werten
        """
        if not all(isinstance(k, int) and k > 0 for k in k_values):
            raise ValueError("K-Werte müssen positive Ganzzahlen sein")
        
        self._config["evaluation"]["k_values"] = k_values
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        Setzt die Batch-Größe für die Evaluierung.
        
        Args:
            batch_size: Batch-Größe
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch-Größe muss eine positive Ganzzahl sein")
        
        self._config["evaluation"]["batch_size"] = batch_size
    
    def set_seed(self, seed: int) -> None:
        """
        Setzt den Seed für Reproduzierbarkeit.
        
        Args:
            seed: Seed-Wert
        """
        if not isinstance(seed, int):
            raise ValueError("Seed muss eine Ganzzahl sein")
        
        self._config["experiment"]["seed"] = seed
    
    def create_variant(self, name: str, changes: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Erstellt eine Variante der Experiment-Konfiguration.
        
        Args:
            name: Name der Variante
            changes: Dictionary mit Änderungen
            
        Returns:
            Neue Experiment-Konfiguration
        """
        new_config = self.to_dict()
        
        # Änderungen anwenden
        for key, value in changes.items():
            if "." in key:
                # Verschachtelte Schlüssel unterstützen
                parts = key.split(".")
                current = new_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                new_config[key] = value
        
        # Experiment-Name aktualisieren
        new_config["experiment"]["name"] = name
        
        return ExperimentConfig(new_config)


def get_default_experiment_config() -> ExperimentConfig:
    """
    Gibt die Standard-Experiment-Konfiguration zurück.
    
    Returns:
        Standard Experiment-Konfiguration
    """
    return ExperimentConfig()


def get_quick_evaluation_config() -> ExperimentConfig:
    """
    Gibt eine Konfiguration für schnelle Evaluierungen zurück.
    
    Returns:
        Schnelle Evaluierungs-Konfiguration
    """
    config = ExperimentConfig()
    
    # Nur grundlegende Metriken aktivieren
    config.set_nested("evaluation.metrics.retrieval", ["precision_at_k", "recall_at_k"])
    config.set_nested("evaluation.metrics.generation", ["rouge_l"])
    config.set_nested("evaluation.metrics.rag", ["faithfulness"])
    config.set_nested("evaluation.metrics.performance", ["inference_time"])
    
    # Weniger K-Werte
    config.set_k_values([1, 5])
    
    # Kleinere Batch-Größe
    config.set_batch_size(16)
    
    # Weniger detaillierte Outputs
    config.set_nested("output.save_detailed_results", False)
    config.set_nested("output.include_visualizations", False)
    
    config.set_nested("experiment.name", "quick_evaluation")
    config.set_nested("experiment.description", "Schnelle Evaluierung mit reduzierten Metriken")
    
    return config


def get_comprehensive_evaluation_config() -> ExperimentConfig:
    """
    Gibt eine Konfiguration für umfassende Evaluierungen zurück.
    
    Returns:
        Umfassende Evaluierungs-Konfiguration
    """
    config = ExperimentConfig()
    
    # Alle Metriken aktivieren
    config.set_nested("evaluation.metrics.retrieval", ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg"])
    config.set_nested("evaluation.metrics.generation", ["rouge_l", "bleu", "bert_score", "semantic_similarity", "exact_match"])
    config.set_nested("evaluation.metrics.rag", ["faithfulness", "groundedness", "ragas_score"])
    config.set_nested("evaluation.metrics.performance", ["inference_time", "memory_usage", "throughput"])
    
    # Mehr K-Werte
    config.set_k_values([1, 3, 5, 10, 20])
    
    # Alle Outputs aktivieren
    config.set_nested("output.save_detailed_results", True)
    config.set_nested("output.include_visualizations", True)
    config.set_nested("output.export_formats", ["json", "csv", "xlsx"])
    
    config.set_nested("experiment.name", "comprehensive_evaluation")
    config.set_nested("experiment.description", "Umfassende Evaluierung mit allen verfügbaren Metriken")
    
    return config 