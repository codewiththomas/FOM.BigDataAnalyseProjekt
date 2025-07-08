from typing import Dict, Any, Optional
from .base_config import BaseConfig


class PipelineConfig(BaseConfig):
    """
    Konfiguration für RAG-Pipeline.

    Definiert alle Parameter für Chunker, Embedding, Vector Store und Language Model.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialisiert die Pipeline-Konfiguration.

        Args:
            config_dict: Optionales Dictionary mit Konfigurationswerten
        """
        # Defaults mit übergebenen Werten zusammenführen
        defaults = self.get_default_config()
        if config_dict:
            defaults.update(config_dict)

        super().__init__(defaults)

    def get_default_config(self) -> Dict[str, Any]:
        """
        Gibt die Standard-Pipeline-Konfiguration zurück.

        Returns:
            Dictionary mit Standard-Konfigurationswerten
        """
        return {
            "chunker": {
                "type": "line_chunker",
                "chunk_size": 512,
                "overlap": 50
            },
            "embedding": {
                "type": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536
            },
            "vector_store": {
                "type": "in_memory",
                "similarity_metric": "cosine",
                "top_k": 5
            },
            "language_model": {
                "type": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 500
            },
            "pipeline": {
                "name": "baseline_pipeline",
                "version": "1.0.0",
                "description": "Baseline RAG Pipeline mit OpenAI-Komponenten"
            }
        }

    def _validate_config(self) -> None:
        """
        Validiert die Pipeline-Konfiguration.
        """
        required_sections = ["chunker", "embedding", "vector_store", "language_model"]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Erforderliche Sektion '{section}' fehlt in der Konfiguration")

        # Chunker-Validierung
        chunker_config = self._config["chunker"]
        if "type" not in chunker_config:
            raise ValueError("Chunker-Typ ist erforderlich")

        if chunker_config.get("chunk_size", 0) <= 0:
            raise ValueError("Chunk-Größe muss positiv sein")

        if chunker_config.get("overlap", 0) < 0:
            raise ValueError("Overlap kann nicht negativ sein")

        # Embedding-Validierung
        embedding_config = self._config["embedding"]
        if "type" not in embedding_config:
            raise ValueError("Embedding-Typ ist erforderlich")

        if embedding_config.get("dimensions", 0) <= 0:
            raise ValueError("Embedding-Dimensionen müssen positiv sein")

        # Vector Store-Validierung
        vector_store_config = self._config["vector_store"]
        if "type" not in vector_store_config:
            raise ValueError("Vector Store-Typ ist erforderlich")

        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if vector_store_config.get("similarity_metric") not in valid_metrics:
            raise ValueError(f"Similarity Metric muss einer von {valid_metrics} sein")

        if vector_store_config.get("top_k", 0) <= 0:
            raise ValueError("Top-K muss positiv sein")

        # Language Model-Validierung
        llm_config = self._config["language_model"]
        if "type" not in llm_config:
            raise ValueError("Language Model-Typ ist erforderlich")

        if not (0 <= llm_config.get("temperature", 0) <= 2):
            raise ValueError("Temperatur muss zwischen 0 und 2 liegen")

        if llm_config.get("max_tokens", 0) <= 0:
            raise ValueError("Max-Tokens muss positiv sein")

    def get_chunker_config(self) -> Dict[str, Any]:
        """
        Gibt die Chunker-Konfiguration zurück.

        Returns:
            Dictionary mit Chunker-Konfiguration
        """
        return self._config["chunker"].copy()

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Gibt die Embedding-Konfiguration zurück.

        Returns:
            Dictionary mit Embedding-Konfiguration
        """
        return self._config["embedding"].copy()

    def get_vector_store_config(self) -> Dict[str, Any]:
        """
        Gibt die Vector Store-Konfiguration zurück.

        Returns:
            Dictionary mit Vector Store-Konfiguration
        """
        return self._config["vector_store"].copy()

    def get_language_model_config(self) -> Dict[str, Any]:
        """
        Gibt die Language Model-Konfiguration zurück.

        Returns:
            Dictionary mit Language Model-Konfiguration
        """
        return self._config["language_model"].copy()

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Gibt Pipeline-Informationen zurück.

        Returns:
            Dictionary mit Pipeline-Informationen
        """
        return self._config.get("pipeline", {}).copy()

    def create_variant(self, component_type: str, component_config: Dict[str, Any]) -> 'PipelineConfig':
        """
        Erstellt eine Variante der Konfiguration mit geänderter Komponente.

        Args:
            component_type: Typ der Komponente ("chunker", "embedding", "vector_store", "language_model")
            component_config: Neue Konfiguration für die Komponente

        Returns:
            Neue Pipeline-Konfiguration
        """
        new_config = self.to_dict()
        new_config[component_type] = component_config

        # Pipeline-Info aktualisieren
        if "pipeline" in new_config:
            new_config["pipeline"]["name"] = f"{new_config['pipeline']['name']}_{component_type}_variant"

        return PipelineConfig(new_config)

    def get_component_types(self) -> Dict[str, str]:
        """
        Gibt die Typen aller Komponenten zurück.

        Returns:
            Dictionary mit Komponententypen
        """
        return {
            "chunker": self._config["chunker"]["type"],
            "embedding": self._config["embedding"]["type"],
            "vector_store": self._config["vector_store"]["type"],
            "language_model": self._config["language_model"]["type"]
        }


def get_baseline_config() -> PipelineConfig:
    """
    Gibt die Baseline-Konfiguration zurück.

    Returns:
        Baseline Pipeline-Konfiguration
    """
    return PipelineConfig()


def get_alternative_configs() -> Dict[str, PipelineConfig]:
    """
    Gibt alternative Konfigurationen für Experimente zurück.

    Returns:
        Dictionary mit alternativen Konfigurationen
    """
    baseline = get_baseline_config()

    configs = {
        "baseline": baseline,

        # Chunker-Varianten
        "recursive_chunker": baseline.create_variant("chunker", {
            "type": "recursive_chunker",
            "chunk_size": 1000,
            "overlap": 100,
            "separators": ["\n\n", "\n", ".", "!", "?"]
        }),

        # Embedding-Varianten
        "sentence_transformer": baseline.create_variant("embedding", {
            "type": "sentence_transformer",
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384
        }),

        # Vector Store-Varianten
        "chroma_store": baseline.create_variant("vector_store", {
            "type": "chroma",
            "similarity_metric": "cosine",
            "top_k": 5,
            "persist_directory": "data/chroma_db"
        }),

        # Language Model-Varianten
        "gpt4_model": baseline.create_variant("language_model", {
            "type": "openai",
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 1000
        })
    }

    return configs


def create_custom_config(
    chunker_type: str = "line_chunker",
    embedding_type: str = "openai",
    vector_store_type: str = "in_memory",
    llm_type: str = "openai",
    **kwargs
) -> PipelineConfig:
    """
    Erstellt eine benutzerdefinierte Konfiguration.

    Args:
        chunker_type: Typ des Chunkers
        embedding_type: Typ des Embeddings
        vector_store_type: Typ des Vector Stores
        llm_type: Typ des Language Models
        **kwargs: Zusätzliche Konfigurationsparameter

    Returns:
        Benutzerdefinierte Pipeline-Konfiguration
    """
    config = {
        "chunker": {"type": chunker_type},
        "embedding": {"type": embedding_type},
        "vector_store": {"type": vector_store_type},
        "language_model": {"type": llm_type},
        "pipeline": {
            "name": "custom_pipeline",
            "version": "1.0.0",
            "description": "Benutzerdefinierte RAG Pipeline"
        }
    }

    # Zusätzliche Parameter hinzufügen
    for key, value in kwargs.items():
        if "." in key:
            # Verschachtelte Schlüssel unterstützen
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value

    return PipelineConfig(config)