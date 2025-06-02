"""
Basisklassen und Interfaces für Retrieval-Komponenten im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ...components.data_sources.base import Document
from ...components.embeddings.base import Embedder
from ...components.vector_stores.base import VectorStore


class Retriever(ABC):
    """
    Basisklasse für alle Retriever.

    Ermöglicht das Abrufen relevanter Dokumente basierend auf einer Query.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3, threshold: Optional[float] = None) -> List[Document]:
        """
        Ruft relevante Dokumente basierend auf einer Query ab.

        Args:
            query: Die Benutzeranfrage
            top_k: Die Anzahl der zurückzugebenden relevantesten Dokumente
            threshold: Optional ein Schwellenwert für die Relevanz

        Returns:
            Eine Liste von relevanten Dokumenten
        """
        pass

    @abstractmethod
    def format_retrieved_documents(self, documents: List[Document]) -> str:
        """
        Formatiert die abgerufenen Dokumente für die Verwendung im RAG-System.

        Args:
            documents: Die abgerufenen Dokumente

        Returns:
            Ein formatierter String, der die Dokumente enthält
        """
        pass