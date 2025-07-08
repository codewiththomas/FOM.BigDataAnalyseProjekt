from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BaseVectorStore(ABC):
    """
    Abstrakte Basisklasse für alle Vector Store-Implementierungen.
    
    Diese Klasse definiert das Interface für verschiedene Vector Store-Implementierungen
    und stellt gemeinsame Funktionalitäten bereit.
    """
    
    def __init__(self, similarity_metric: str = "cosine", **kwargs):
        """
        Initialisiert den Vector Store.
        
        Args:
            similarity_metric: Ähnlichkeitsmetrik für die Suche
            **kwargs: Weitere store-spezifische Parameter
        """
        self.similarity_metric = similarity_metric
        self.config = kwargs
        self._embeddings = []
        self._texts = []
        self._metadata = []
    
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray, 
                  metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Fügt Texte mit ihren Embeddings zum Vector Store hinzu.
        
        Args:
            texts: Liste von Texten
            embeddings: Embeddings der Texte
            metadata: Optionale Metadaten für jeden Text
            
        Returns:
            Liste von IDs der hinzugefügten Texte
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sucht nach ähnlichen Texten basierend auf einem Query-Embedding.
        
        Args:
            query_embedding: Embedding der Query
            top_k: Anzahl der zurückzugebenden Ergebnisse
            
        Returns:
            Liste von Dictionaries mit Text, Score und Metadaten
        """
        pass
    
    @abstractmethod
    def delete_texts(self, ids: List[str]) -> bool:
        """
        Löscht Texte aus dem Vector Store.
        
        Args:
            ids: Liste von IDs der zu löschenden Texte
            
        Returns:
            True wenn erfolgreich gelöscht
        """
        pass
    
    @abstractmethod
    def update_texts(self, ids: List[str], texts: List[str], 
                     embeddings: np.ndarray, 
                     metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Aktualisiert existierende Texte im Vector Store.
        
        Args:
            ids: Liste von IDs der zu aktualisierenden Texte
            texts: Neue Texte
            embeddings: Neue Embeddings
            metadata: Neue Metadaten
            
        Returns:
            True wenn erfolgreich aktualisiert
        """
        pass
    
    def similarity_search_with_score(self, query_embedding: np.ndarray, 
                                   top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Sucht nach ähnlichen Texten und gibt Scores zurück.
        
        Args:
            query_embedding: Embedding der Query
            top_k: Anzahl der zurückzugebenden Ergebnisse
            
        Returns:
            Liste von Tupeln (Ergebnis, Score)
        """
        results = self.similarity_search(query_embedding, top_k)
        return [(result, result.get("score", 0.0)) for result in results]
    
    def get_text_by_id(self, text_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt einen Text anhand seiner ID zurück.
        
        Args:
            text_id: ID des Textes
            
        Returns:
            Dictionary mit Text und Metadaten oder None
        """
        # Standardimplementierung - kann in Subklassen überschrieben werden
        return None
    
    def get_all_texts(self) -> List[Dict[str, Any]]:
        """
        Gibt alle Texte im Vector Store zurück.
        
        Returns:
            Liste aller Texte mit Metadaten
        """
        return [{"text": text, "metadata": meta} 
                for text, meta in zip(self._texts, self._metadata)]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den Vector Store zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        return {
            "total_texts": len(self._texts),
            "similarity_metric": self.similarity_metric,
            "store_type": self.__class__.__name__
        }
    
    def clear(self) -> bool:
        """
        Löscht alle Daten aus dem Vector Store.
        
        Returns:
            True wenn erfolgreich geleert
        """
        self._embeddings = []
        self._texts = []
        self._metadata = []
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration des Vector Stores zurück.
        
        Returns:
            Dictionary mit Konfigurationsparametern
        """
        return {
            "type": self.__class__.__name__,
            "similarity_metric": self.similarity_metric,
            **self.config
        }
    
    def __len__(self) -> int:
        return len(self._texts)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(metric={self.similarity_metric}, size={len(self)})"
    
    def __repr__(self) -> str:
        return self.__str__() 