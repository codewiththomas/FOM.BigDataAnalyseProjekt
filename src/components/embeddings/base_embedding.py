from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np


class BaseEmbedding(ABC):
    """
    Abstrakte Basisklasse für alle Embedding-Strategien.
    
    Diese Klasse definiert das Interface für verschiedene Embedding-Implementierungen
    und stellt gemeinsame Funktionalitäten bereit.
    """
    
    def __init__(self, model_name: str, dimensions: Optional[int] = None, **kwargs):
        """
        Initialisiert das Embedding-Modell.
        
        Args:
            model_name: Name des Embedding-Modells
            dimensions: Dimensionalität der Embeddings
            **kwargs: Weitere modell-spezifische Parameter
        """
        self.model_name = model_name
        self.dimensions = dimensions
        self.config = kwargs
        self._model = None
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Erstellt Embeddings für eine Liste von Texten.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            NumPy-Array mit Embeddings (shape: [len(texts), dimensions])
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Erstellt Embedding für eine einzelne Query.
        
        Args:
            query: Query-Text
            
        Returns:
            NumPy-Array mit Embedding (shape: [dimensions])
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Lädt das Embedding-Modell.
        """
        pass
    
    def batch_embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Erstellt Embeddings für große Textmengen in Batches.
        
        Args:
            texts: Liste von Texten
            batch_size: Größe der Batches
            
        Returns:
            NumPy-Array mit allen Embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Gibt die Dimensionalität der Embeddings zurück.
        
        Returns:
            Anzahl der Dimensionen
        """
        if self.dimensions is not None:
            return self.dimensions
        
        # Fallback: Teste mit einem kleinen Text
        test_embedding = self.embed_query("test")
        return test_embedding.shape[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                  metric: str = "cosine") -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Embeddings.
        
        Args:
            embedding1: Erstes Embedding
            embedding2: Zweites Embedding
            metric: Ähnlichkeitsmetrik ("cosine", "euclidean", "dot_product")
            
        Returns:
            Ähnlichkeitswert
        """
        if metric == "cosine":
            return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == "euclidean":
            return -np.linalg.norm(embedding1 - embedding2)
        elif metric == "dot_product":
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unbekannte Ähnlichkeitsmetrik: {metric}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration des Embedding-Modells zurück.
        
        Returns:
            Dictionary mit Konfigurationsparametern
        """
        return {
            "type": self.__class__.__name__,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            **self.config
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.dimensions})"
    
    def __repr__(self) -> str:
        return self.__str__() 