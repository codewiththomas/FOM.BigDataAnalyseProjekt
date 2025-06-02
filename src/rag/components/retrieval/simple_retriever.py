"""
Implementiert einen einfachen Retriever für das RAG-System.
"""

import logging
from typing import List, Optional

from ...config import config
from ...components.data_sources.base import Document
from ...components.embeddings.base import Embedder
from ...components.vector_stores.base import VectorStore
from .base import Retriever


class SimpleRetriever(Retriever):
    """
    Implementiert einen einfachen Retriever für das RAG-System.

    Dieser Retriever nutzt einen Vector Store, um relevante Dokumente abzurufen.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 4,
        threshold: Optional[float] = None
    ):
        """
        Initialisiert den SimpleRetriever.

        Args:
            vector_store: Vector Store für die Dokumente
            embedder: Embedder für die Queries
            top_k: Standardanzahl der zurückzugebenden Dokumente
            threshold: Standardschwellenwert für die Relevanz
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold

        # Logger für Debug-Informationen
        self.logger = logging.getLogger(__name__)

        # Debug-Ausgabe: Aktuellen Schwellenwert und Top-K anzeigen
        print(f"[DEBUG] Retriever initialisiert mit top_k={self.top_k}, threshold={self.threshold}")

    def retrieve(self, query: str, top_k: int = None, threshold: Optional[float] = None) -> List[Document]:
        """
        Ruft relevante Dokumente basierend auf einer Query ab.

        Args:
            query: Die Benutzeranfrage
            top_k: Die Anzahl der zurückzugebenden relevantesten Dokumente
            threshold: Optional ein Schwellenwert für die Relevanz

        Returns:
            Eine Liste von relevanten Dokumenten
        """
        # Fallback auf Standardwerte, wenn keine Parameter angegeben wurden
        top_k = top_k or self.top_k
        threshold = threshold if threshold is not None else self.threshold

        # Debug-Ausgabe: Zeige die Query
        print(f"[DEBUG] Retrieval für Query: '{query}'")
        print(f"[DEBUG] Verwende top_k={top_k}, threshold={threshold}")

        # Query embedden
        query_embedding = self.embedder.embed_query(query)

        # Debug-Ausgabe: Dimension des Query-Embeddings
        print(f"[DEBUG] Query-Embedding Dimension: {len(query_embedding)}")

        # Relevante Dokumente abrufen
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            threshold=threshold
        )

        # Debug-Ausgabe: Anzahl der gefundenen Dokumente und Ähnlichkeitswerte
        print(f"[DEBUG] Gefundene Dokumente: {len(results)}")
        for i, (doc, score) in enumerate(results):
            print(f"[DEBUG] Dokument {i+1}: Score={score:.4f}, Content={doc.text[:50]}...")

        # Konvertiere die Vector Store Documents in Source Documents
        documents = []
        for doc, score in results:
            source_doc = Document(
                content=doc.text,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            documents.append(source_doc)

        return documents

    def format_retrieved_documents(self, documents: List[Document]) -> str:
        """
        Formatiert die abgerufenen Dokumente für die Verwendung im RAG-System.

        Args:
            documents: Die abgerufenen Dokumente

        Returns:
            Ein formatierter String, der die Dokumente enthält
        """
        if not documents:
            return "Es wurden keine relevanten Dokumente gefunden."

        formatted_texts = []

        for i, doc in enumerate(documents):
            source_info = f"Quelle: {doc.metadata.get('source', 'Unbekannt')}" if doc.metadata else "Quelle: Unbekannt"
            chunk_info = f"Chunk: {doc.metadata.get('chunk', 'Unbekannt')}/{doc.metadata.get('chunk_count', 'Unbekannt')}" if doc.metadata else ""
            formatted_texts.append(f"[Dokument {i+1}] {source_info} {chunk_info}\n{doc.content}\n")

        result = "\n".join(formatted_texts)

        # Debug-Ausgabe: Formatierter Kontext
        print(f"[DEBUG] Formatierter Kontext für LLM:\n{result}")

        return result