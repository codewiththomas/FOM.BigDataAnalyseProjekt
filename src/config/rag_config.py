from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:

    chunker_type: str = "line"
    chunker_params: Dict[str, Any] = None

    embedding_type: str = "sentence_transformers"
    embedding_params: Dict[str, Any] = None

    vector_store_type: str = "in_memory"
    vector_store_params: Dict[str, Any] = None

    language_model_type: str = "openai"
    language_model_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.chunker_params is None:
            self.chunker_params = {}
        if self.embedding_params is None:
            self.embedding_params = {}
        if self.vector_store_params is None:
            self.vector_store_params = {}
        if self.language_model_params is None:
            self.language_model_params = {}