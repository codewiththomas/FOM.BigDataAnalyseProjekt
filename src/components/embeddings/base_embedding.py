from abc import ABC, abstractmethod
from typing import List

class BaseEmbedding(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Erstellt Embedding für einen Text"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Erstellt Embeddings für mehrere Texte"""
        pass