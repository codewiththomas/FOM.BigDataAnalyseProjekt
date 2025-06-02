"""
Implementierung des Vector Stores mit Qdrant.
"""

from typing import List, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models

from src.rag.components.vector_stores.base import BaseVectorStore, Document
from src.rag.config import config


class QdrantVectorStore(BaseVectorStore):
    """Vector Store Implementierung mit Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "documents",
        vector_size: int = 1536  # Standard für OpenAI Embeddings
    ):
        """
        Initialisiert den Qdrant Vector Store.
        
        Args:
            client: Qdrant Client Instanz
            collection_name: Name der Collection
            vector_size: Größe der Embedding-Vektoren
        """
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Collection erstellen, falls sie nicht existiert
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self) -> None:
        """Erstellt die Collection, falls sie nicht existiert."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Fügt Dokumente zum Vector Store hinzu.
        
        Args:
            documents: Liste von Dokumenten zum Hinzufügen
        """
        if not documents:
            return

        # Vorbereiten der Punkte für Qdrant
        points = []
        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(f"Expected Document object, got {type(doc)}")
            
            if not hasattr(doc, 'embedding') or doc.embedding is None:
                raise ValueError(f"Document at index {idx} has no embedding")
                
            if not hasattr(doc, 'text') or doc.text is None:
                raise ValueError(f"Document at index {idx} has no text")

            point = models.PointStruct(
                id=idx,
                vector=doc.embedding.tolist(),
                payload={
                    "text": doc.text,
                    "metadata": doc.metadata if doc.metadata else {}
                }
            )
            points.append(point)

        # Punkte zur Collection hinzufügen
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 4,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Führt eine Ähnlichkeitssuche durch.
        
        Args:
            query_embedding: Embedding des Queries
            k: Anzahl der zurückzugebenden Dokumente
            threshold: Optionaler Schwellenwert für die Ähnlichkeit
        
        Returns:
            Liste von Tupeln (Document, Score)
        """
        # Suche in Qdrant durchführen
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            score_threshold=threshold if threshold is not None else None
        )

        # Ergebnisse in Document-Objekte umwandeln
        results = []
        for hit in search_result:
            doc = Document(
                text=hit.payload["text"],
                embedding=np.array(hit.vector),
                metadata=hit.payload.get("metadata", {})
            )
            results.append((doc, hit.score))

        return results

    def clear(self) -> None:
        """Löscht alle Dokumente aus dem Vector Store."""
        try:
            self.client.delete_collection(self.collection_name)
            self._create_collection_if_not_exists()
        except Exception as e:
            print(f"Fehler beim Löschen der Collection: {e}")

    @property
    def document_count(self) -> int:
        """
        Gibt die Anzahl der im Vector Store gespeicherten Dokumente zurück.
        
        Returns:
            Anzahl der Dokumente
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"Fehler beim Abrufen der Dokumentenanzahl: {e}")
            return 0 