from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: List[List[float]], 
                  metadatas: List[Dict[str, Any]] = None):
        """Fügt Texte mit Embeddings hinzu"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Sucht ähnliche Texte"""
        pass 