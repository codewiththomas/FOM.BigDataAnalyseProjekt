from typing import List, Dict, Any, Optional
import numpy as np
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from .base_vector_store import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    In-Memory Vector Store-Implementierung als Baseline.
    
    Speichert alle Embeddings im Arbeitsspeicher und nutzt sklearn für Ähnlichkeitssuche.
    """
    
    def __init__(self, similarity_metric: str = "cosine", **kwargs):
        """
        Initialisiert den In-Memory Vector Store.
        
        Args:
            similarity_metric: Ähnlichkeitsmetrik ("cosine", "euclidean", "dot_product")
            **kwargs: Weitere Parameter
        """
        super().__init__(similarity_metric, **kwargs)
        self._embeddings = []
        self._texts = []
        self._metadata = []
        self._ids = []
    
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
        if len(texts) != len(embeddings):
            raise ValueError("Anzahl der Texte muss der Anzahl der Embeddings entsprechen")
        
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Anzahl der Metadaten muss der Anzahl der Texte entsprechen")
        
        # IDs generieren
        new_ids = [str(uuid.uuid4()) for _ in texts]
        
        # Daten hinzufügen
        self._ids.extend(new_ids)
        self._texts.extend(texts)
        self._embeddings.extend(embeddings)
        
        if metadata is not None:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])
        
        return new_ids
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sucht nach ähnlichen Texten basierend auf einem Query-Embedding.
        
        Args:
            query_embedding: Embedding der Query
            top_k: Anzahl der zurückzugebenden Ergebnisse
            
        Returns:
            Liste von Dictionaries mit Text, Score und Metadaten
        """
        if not self._embeddings:
            return []
        
        # Embeddings zu NumPy-Array konvertieren
        embeddings_array = np.array(self._embeddings)
        
        # Ähnlichkeiten berechnen
        if self.similarity_metric == "cosine":
            similarities = cosine_similarity([query_embedding], embeddings_array)[0]
        elif self.similarity_metric == "euclidean":
            # Negative Distanz für Ähnlichkeit (höher = ähnlicher)
            distances = np.linalg.norm(embeddings_array - query_embedding, axis=1)
            similarities = -distances
        elif self.similarity_metric == "dot_product":
            similarities = np.dot(embeddings_array, query_embedding)
        else:
            raise ValueError(f"Unbekannte Ähnlichkeitsmetrik: {self.similarity_metric}")
        
        # Top-K Ergebnisse finden
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = {
                "id": self._ids[idx],
                "text": self._texts[idx],
                "score": float(similarities[idx]),
                "metadata": self._metadata[idx].copy()
            }
            results.append(result)
        
        return results
    
    def delete_texts(self, ids: List[str]) -> bool:
        """
        Löscht Texte aus dem Vector Store.
        
        Args:
            ids: Liste von IDs der zu löschenden Texte
            
        Returns:
            True wenn erfolgreich gelöscht
        """
        try:
            indices_to_remove = []
            
            for id_to_remove in ids:
                try:
                    idx = self._ids.index(id_to_remove)
                    indices_to_remove.append(idx)
                except ValueError:
                    # ID nicht gefunden, ignorieren
                    pass
            
            # Indices sortieren (absteigend) um korrekte Löschung zu gewährleisten
            indices_to_remove.sort(reverse=True)
            
            for idx in indices_to_remove:
                del self._ids[idx]
                del self._texts[idx]
                del self._embeddings[idx]
                del self._metadata[idx]
            
            return True
            
        except Exception:
            return False
    
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
        if len(ids) != len(texts) or len(texts) != len(embeddings):
            return False
        
        if metadata is not None and len(metadata) != len(texts):
            return False
        
        try:
            for i, id_to_update in enumerate(ids):
                try:
                    idx = self._ids.index(id_to_update)
                    self._texts[idx] = texts[i]
                    self._embeddings[idx] = embeddings[i]
                    
                    if metadata is not None:
                        self._metadata[idx] = metadata[i]
                        
                except ValueError:
                    # ID nicht gefunden, ignorieren
                    pass
            
            return True
            
        except Exception:
            return False
    
    def get_text_by_id(self, text_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt einen Text anhand seiner ID zurück.
        
        Args:
            text_id: ID des Textes
            
        Returns:
            Dictionary mit Text und Metadaten oder None
        """
        try:
            idx = self._ids.index(text_id)
            return {
                "id": text_id,
                "text": self._texts[idx],
                "metadata": self._metadata[idx].copy()
            }
        except ValueError:
            return None
    
    def get_all_texts(self) -> List[Dict[str, Any]]:
        """
        Gibt alle Texte im Vector Store zurück.
        
        Returns:
            Liste aller Texte mit Metadaten
        """
        return [
            {
                "id": self._ids[i],
                "text": self._texts[i],
                "metadata": self._metadata[i].copy()
            }
            for i in range(len(self._texts))
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den Vector Store zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        stats = super().get_stats()
        
        if self._embeddings:
            embedding_dim = len(self._embeddings[0])
            stats.update({
                "embedding_dimension": embedding_dim,
                "memory_usage_mb": self._estimate_memory_usage()
            })
        
        return stats
    
    def clear(self) -> bool:
        """
        Löscht alle Daten aus dem Vector Store.
        
        Returns:
            True wenn erfolgreich geleert
        """
        self._embeddings = []
        self._texts = []
        self._metadata = []
        self._ids = []
        return True
    
    def _estimate_memory_usage(self) -> float:
        """
        Schätzt die Speichernutzung in MB.
        
        Returns:
            Geschätzte Speichernutzung in MB
        """
        if not self._embeddings:
            return 0.0
        
        # Grobe Schätzung
        embeddings_size = len(self._embeddings) * len(self._embeddings[0]) * 8  # 8 Bytes pro float64
        texts_size = sum(len(text.encode('utf-8')) for text in self._texts)
        metadata_size = len(str(self._metadata).encode('utf-8'))
        
        total_bytes = embeddings_size + texts_size + metadata_size
        return total_bytes / (1024 * 1024)  # Konvertierung zu MB 