from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseChunker(ABC):
    """
    Abstrakte Basisklasse für alle Chunking-Strategien.
    
    Diese Klasse definiert das Interface für verschiedene Chunking-Implementierungen
    und stellt gemeinsame Funktionalitäten bereit.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, **kwargs):
        """
        Initialisiert den Chunker.
        
        Args:
            chunk_size: Maximale Größe eines Chunks in Zeichen
            overlap: Überlappung zwischen Chunks in Zeichen
            **kwargs: Weitere chunker-spezifische Parameter
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.config = kwargs
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        Teilt einen Text in Chunks auf.
        
        Args:
            text: Der zu teilende Text
            
        Returns:
            Liste von Text-Chunks
        """
        pass
    
    def chunk_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Teilt mehrere Dokumente in Chunks auf.
        
        Args:
            documents: Liste von Dokumenten
            
        Returns:
            Liste von Chunk-Dictionaries mit Metadaten
        """
        all_chunks = []
        
        for doc_id, document in enumerate(documents):
            chunks = self.chunk_text(document)
            
            for chunk_id, chunk in enumerate(chunks):
                chunk_data = {
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                    "metadata": {
                        "chunker_type": self.__class__.__name__,
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap
                    }
                }
                all_chunks.append(chunk_data)
        
        return all_chunks
    
    def get_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration des Chunkers zurück.
        
        Returns:
            Dictionary mit Konfigurationsparametern
        """
        return {
            "type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            **self.config
        }
    
    def validate_chunk_size(self, text: str) -> bool:
        """
        Validiert, ob die Chunk-Größe für den gegebenen Text sinnvoll ist.
        
        Args:
            text: Zu validierender Text
            
        Returns:
            True wenn die Chunk-Größe sinnvoll ist
        """
        return len(text) > self.chunk_size or self.chunk_size > 0
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, overlap={self.overlap})"
    
    def __repr__(self) -> str:
        return self.__str__() 