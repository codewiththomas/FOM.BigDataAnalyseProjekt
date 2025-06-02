"""
Implementiert einen In-Memory Vector Store für das RAG-System.
"""

import numpy as np
from typing import List, Optional, Tuple
from enum import Enum

from ...components.data_sources.base import Document
from .base import VectorStore


class SimilarityMetric(str, Enum):
    """
    Verfügbare Ähnlichkeitsmetriken für den Vector Store.
    """
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class InMemoryVectorStore(VectorStore):
    """
    Implementiert einen einfachen In-Memory Vector Store.

    Dieser Vector Store speichert Embeddings und Dokumente im Arbeitsspeicher.
    """

    def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric.COSINE):
        """
        Initialisiert den InMemoryVectorStore.

        Args:
            similarity_metric: Die zu verwendende Ähnlichkeitsmetrik
        """
        self.documents = []
        self.embeddings = []
        self.similarity_metric = similarity_metric

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Fügt Dokumente und ihre Embeddings zum Vector Store hinzu.

        Args:
            documents: Eine Liste von Document-Objekten
            embeddings: Eine Liste von Embedding-Vektoren
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Anzahl der Dokumente ({len(documents)}) stimmt nicht mit der Anzahl der Embeddings ({len(embeddings)}) überein."
            )

        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Führt eine Ähnlichkeitssuche im Vector Store durch.

        Args:
            query_embedding: Das Embedding der Query
            k: Die Anzahl der zurückzugebenden ähnlichsten Dokumente
            threshold: Optional ein Schwellenwert für die Ähnlichkeit

        Returns:
            Eine Liste von Tupeln (Document, Ähnlichkeitswert)
        """
        if not self.embeddings:
            return []

        # Query-Embedding in NumPy-Array umwandeln
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        # Alle Embeddings in NumPy-Array umwandeln
        embeddings_np = np.array(self.embeddings, dtype=np.float32)

        # Ähnlichkeit berechnen
        if self.similarity_metric == SimilarityMetric.COSINE:
            # Normalisieren
            query_norm = np.linalg.norm(query_embedding_np)
            if query_norm > 0:
                query_embedding_np = query_embedding_np / query_norm

            # Embeddings normalisieren, falls nicht bereits normalisiert
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Division durch 0 vermeiden
            embeddings_np = embeddings_np / norms

            # Kosinus-Ähnlichkeit berechnen (höher ist besser)
            similarities = np.dot(embeddings_np, query_embedding_np)

        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            # Skalarprodukt berechnen (höher ist besser)
            similarities = np.dot(embeddings_np, query_embedding_np)

        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            # Euklidische Distanz berechnen (niedriger ist besser)
            distances = np.linalg.norm(embeddings_np - query_embedding_np, axis=1)
            # In Ähnlichkeit umwandeln (1 / (1 + Distanz))
            similarities = 1 / (1 + distances)

        else:
            raise ValueError(f"Unbekannte Ähnlichkeitsmetrik: {self.similarity_metric}")

        # Ergebnisse filtern, falls ein Schwellenwert angegeben wurde
        if threshold is not None:
            mask = similarities >= threshold
            indices = np.nonzero(mask)[0]

            # Sortierte Indizes der ähnlichsten Dokumente (absteigend)
            sorted_indices = indices[np.argsort(-similarities[indices])]

            # Auf k begrenzen
            sorted_indices = sorted_indices[:k]

            # Ergebnisse zusammenstellen
            results = [(self.documents[i], float(similarities[i])) for i in sorted_indices]
        else:
            # Top-k ähnlichste Dokumente finden
            sorted_indices = np.argsort(-similarities)[:k]

            # Ergebnisse zusammenstellen
            results = [(self.documents[i], float(similarities[i])) for i in sorted_indices]

        return results

    def clear(self) -> None:
        """Löscht alle Dokumente und Embeddings aus dem Vector Store."""
        self.documents = []
        self.embeddings = []

    @property
    def document_count(self) -> int:
        """
        Gibt die Anzahl der im Vector Store gespeicherten Dokumente zurück.

        Returns:
            Anzahl der Dokumente
        """
        return len(self.documents)