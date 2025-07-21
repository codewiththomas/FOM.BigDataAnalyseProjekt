from typing import Dict, Any, Type, Optional, Union
import logging
import importlib
from components.chunkers import BaseChunker, LineChunker
from components.embeddings import BaseEmbedding, OpenAIEmbedding
from components.vector_stores import BaseVectorStore, InMemoryVectorStore
from components.language_models import BaseLanguageModel, OpenAILanguageModel
from evaluations import (
    BaseEvaluator, RetrievalEvaluator, GenerationEvaluator,
    RAGEvaluator, PerformanceEvaluator
)


class ComponentLoader:
    """
    Lädt Komponenten dynamisch basierend auf Konfiguration.

    Ermöglicht das einfache Austauschen von Komponenten ohne Code-Änderungen.
    """

    def __init__(self):
        """
        Initialisiert den Component Loader.
        """
        # Alle verfügbaren Komponenten importieren
        try:
            from components.chunkers import RecursiveChunker, SemanticChunker
            from components.embeddings import SentenceTransformerEmbedding
            from components.language_models import OllamaLanguageModel
            from components.vector_stores import ChromaVectorStore, FAISSVectorStore
        except ImportError as e:
            print(f"Warning: Einige Komponenten nicht verfügbar: {e}")

        self._chunker_registry = {
            "line_chunker": LineChunker,
            "recursive_chunker": RecursiveChunker,
            "semantic_chunker": SemanticChunker,
        }

        self._embedding_registry = {
            "openai": OpenAIEmbedding,
            "sentence_transformer": SentenceTransformerEmbedding,
        }

        self._vector_store_registry = {
            "in_memory": InMemoryVectorStore,
            "chroma": ChromaVectorStore,
            "faiss": FAISSVectorStore,
        }

        self._language_model_registry = {
            "openai": OpenAILanguageModel,
            "ollama": OllamaLanguageModel,
        }

        self._evaluator_registry = {
            "retrieval": RetrievalEvaluator,
            "generation": GenerationEvaluator,
            "rag": RAGEvaluator,
            "performance": PerformanceEvaluator,
        }

    def load_chunker(self, config: Dict[str, Any]) -> BaseChunker:
        """
        Lädt einen Chunker basierend auf der Konfiguration.

        Args:
            config: Chunker-Konfiguration

        Returns:
            Chunker-Instanz
        """
        chunker_type = config.get("type")

        if chunker_type not in self._chunker_registry:
            raise ValueError(f"Unbekannter Chunker-Typ: {chunker_type}")

        chunker_class = self._chunker_registry[chunker_type]

        # Typ aus Konfiguration entfernen für Instanziierung
        init_config = {k: v for k, v in config.items() if k != "type"}

        try:
            return chunker_class(**init_config)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Chunkers '{chunker_type}': {e}")

    def load_embedding(self, config: Dict[str, Any]) -> BaseEmbedding:
        """
        Lädt ein Embedding-Modell basierend auf der Konfiguration.

        Args:
            config: Embedding-Konfiguration

        Returns:
            Embedding-Instanz
        """
        embedding_type = config.get("type")

        if embedding_type not in self._embedding_registry:
            raise ValueError(f"Unbekannter Embedding-Typ: {embedding_type}")

        embedding_class = self._embedding_registry[embedding_type]

        # Typ aus Konfiguration entfernen für Instanziierung
        init_config = {k: v for k, v in config.items() if k != "type"}

        # Spezielle Behandlung für OpenAI Embedding
        if embedding_type == "openai":
            # 'model' zu 'model_name' umbenennen falls vorhanden
            if "model" in init_config:
                init_config["model_name"] = init_config.pop("model")

        try:
            return embedding_class(**init_config)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Embeddings '{embedding_type}': {e}")

    def load_vector_store(self, config: Dict[str, Any]) -> BaseVectorStore:
        """
        Lädt einen Vector Store basierend auf der Konfiguration.

        Args:
            config: Vector Store-Konfiguration

        Returns:
            Vector Store-Instanz
        """
        vector_store_type = config.get("type")

        if vector_store_type not in self._vector_store_registry:
            raise ValueError(f"Unbekannter Vector Store-Typ: {vector_store_type}")

        vector_store_class = self._vector_store_registry[vector_store_type]

        # Typ aus Konfiguration entfernen für Instanziierung
        init_config = {k: v for k, v in config.items() if k != "type"}

        try:
            return vector_store_class(**init_config)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Vector Stores '{vector_store_type}': {e}")

    def load_language_model(self, config: Dict[str, Any]) -> BaseLanguageModel:
        """
        Lädt ein Language Model basierend auf der Konfiguration.

        Args:
            config: Language Model-Konfiguration

        Returns:
            Language Model-Instanz
        """
        llm_type = config.get("type")

        if llm_type not in self._language_model_registry:
            raise ValueError(f"Unbekannter Language Model-Typ: {llm_type}")

        llm_class = self._language_model_registry[llm_type]

        # Typ aus Konfiguration entfernen für Instanziierung
        init_config = {k: v for k, v in config.items() if k != "type"}

        # Spezielle Behandlung für OpenAI Language Model
        if llm_type == "openai":
            # 'model' zu 'model_name' umbenennen falls vorhanden
            if "model" in init_config:
                init_config["model_name"] = init_config.pop("model")

        try:
            return llm_class(**init_config)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Language Models '{llm_type}': {e}")

    def register_chunker(self, name: str, chunker_class: type) -> None:
        """
        Registriert einen neuen Chunker-Typ.

        Args:
            name: Name des Chunker-Typs
            chunker_class: Chunker-Klasse
        """
        if not issubclass(chunker_class, BaseChunker):
            raise ValueError("Chunker-Klasse muss von BaseChunker erben")

        self._chunker_registry[name] = chunker_class

    def register_embedding(self, name: str, embedding_class: type) -> None:
        """
        Registriert einen neuen Embedding-Typ.

        Args:
            name: Name des Embedding-Typs
            embedding_class: Embedding-Klasse
        """
        if not issubclass(embedding_class, BaseEmbedding):
            raise ValueError("Embedding-Klasse muss von BaseEmbedding erben")

        self._embedding_registry[name] = embedding_class

    def register_vector_store(self, name: str, vector_store_class: type) -> None:
        """
        Registriert einen neuen Vector Store-Typ.

        Args:
            name: Name des Vector Store-Typs
            vector_store_class: Vector Store-Klasse
        """
        if not issubclass(vector_store_class, BaseVectorStore):
            raise ValueError("Vector Store-Klasse muss von BaseVectorStore erben")

        self._vector_store_registry[name] = vector_store_class

    def register_language_model(self, name: str, llm_class: type) -> None:
        """
        Registriert einen neuen Language Model-Typ.

        Args:
            name: Name des Language Model-Typs
            llm_class: Language Model-Klasse
        """
        if not issubclass(llm_class, BaseLanguageModel):
            raise ValueError("Language Model-Klasse muss von BaseLanguageModel erben")

        self._language_model_registry[name] = llm_class

    def load_evaluator(self, config: Dict[str, Any]) -> BaseEvaluator:
        """
        Lädt einen Evaluator basierend auf der Konfiguration.

        Args:
            config: Evaluator-Konfiguration

        Returns:
            Evaluator-Instanz
        """
        evaluator_type = config.get("type")

        if evaluator_type not in self._evaluator_registry:
            raise ValueError(f"Unbekannter Evaluator-Typ: {evaluator_type}")

        evaluator_class = self._evaluator_registry[evaluator_type]

        # Typ aus Konfiguration entfernen für Instanziierung
        init_config = {k: v for k, v in config.items() if k != "type"}

        try:
            return evaluator_class(**init_config)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Evaluators '{evaluator_type}': {e}")

    def register_evaluator(self, name: str, evaluator_class: type) -> None:
        """
        Registriert einen neuen Evaluator-Typ.

        Args:
            name: Name des Evaluator-Typs
            evaluator_class: Evaluator-Klasse
        """
        if not issubclass(evaluator_class, BaseEvaluator):
            raise ValueError("Evaluator-Klasse muss von BaseEvaluator erben")

        self._evaluator_registry[name] = evaluator_class

    def get_available_components(self) -> Dict[str, list]:
        """
        Gibt alle verfügbaren Komponententypen zurück.

        Returns:
            Dictionary mit verfügbaren Komponenten
        """
        return {
            "chunkers": list(self._chunker_registry.keys()),
            "embeddings": list(self._embedding_registry.keys()),
            "vector_stores": list(self._vector_store_registry.keys()),
            "language_models": list(self._language_model_registry.keys()),
            "evaluators": list(self._evaluator_registry.keys())
        }

    def load_components_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lädt alle Komponenten basierend auf einer Pipeline-Konfiguration.

        Args:
            config: Pipeline-Konfiguration

        Returns:
            Dictionary mit geladenen Komponenten
        """
        components = {}

        if "chunker" in config:
            components["chunker"] = self.load_chunker(config["chunker"])

        if "embedding" in config:
            components["embedding"] = self.load_embedding(config["embedding"])

        if "vector_store" in config:
            components["vector_store"] = self.load_vector_store(config["vector_store"])

        if "language_model" in config:
            components["language_model"] = self.load_language_model(config["language_model"])

        if "evaluator" in config:
            components["evaluator"] = self.load_evaluator(config["evaluator"])

        return components


# Globale Instanz des Component Loaders
component_loader = ComponentLoader()


def load_component(component_type: str, config: Dict[str, Any]) -> Union[BaseChunker, BaseEmbedding, BaseVectorStore, BaseLanguageModel, BaseEvaluator]:
    """
    Convenience-Funktion zum Laden einer Komponente.

    Args:
        component_type: Typ der Komponente ("chunker", "embedding", "vector_store", "language_model", "evaluator")
        config: Konfiguration der Komponente

    Returns:
        Geladene Komponente
    """
    if component_type == "chunker":
        return component_loader.load_chunker(config)
    elif component_type == "embedding":
        return component_loader.load_embedding(config)
    elif component_type == "vector_store":
        return component_loader.load_vector_store(config)
    elif component_type == "language_model":
        return component_loader.load_language_model(config)
    elif component_type == "evaluator":
        return component_loader.load_evaluator(config)
    else:
        raise ValueError(f"Unbekannter Komponententyp: {component_type}")