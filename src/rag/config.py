"""
Konfigurationsmodul für das RAG-System.
Lädt Umgebungsvariablen und stellt sie als Konfigurationsobjekte zur Verfügung.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()


@dataclass
class LLMConfig:
    """Konfiguration für das Language Model."""
    model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass
class EmbeddingConfig:
    """Konfiguration für den Embedding-Mechanismus."""
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass
class VectorStoreConfig:
    """Konfiguration für den Vector Store."""
    store_type: Literal["in_memory", "qdrant"] = os.getenv("VECTOR_STORE_TYPE", "in_memory")
    similarity: Literal["cosine", "dot_product", "euclidean"] = os.getenv(
        "VECTOR_STORE_SIMILARITY", "cosine")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "documents")


@dataclass
class RetrievalConfig:
    """Konfiguration für den Retrieval-Mechanismus."""
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))  # Erhöht von 3 auf 5
    threshold: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.3"))  # Reduziert von 0.7 auf 0.3


@dataclass
class WebConfig:
    """Konfiguration für die Web-App."""
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "0") == "1"  # Debug-Modus standardmäßig deaktiviert


@dataclass
class AppConfig:
    """Gesamtkonfiguration der Anwendung."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    web: WebConfig = field(default_factory=WebConfig)
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))

    def validate(self) -> bool:
        """Überprüft, ob die Konfiguration gültig ist."""
        if not self.llm.api_key:
            raise ValueError("OpenAI API-Key ist nicht gesetzt. Bitte setze die Umgebungsvariable OPENAI_API_KEY.")
        return True


# Singleton-Instanz für die Konfiguration
config = AppConfig()