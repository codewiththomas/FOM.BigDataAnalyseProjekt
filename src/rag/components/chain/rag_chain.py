"""
Implementiert eine RAG-Chain für das RAG-System.
"""

from typing import Dict, List, Optional, Union, Any, Generator

from ...components.retrieval.base import Retriever
from ...components.llm.base import LLM
from .base import RAGChain


class SimpleRAGChain(RAGChain):
    """
    Implementiert eine einfache RAG-Chain für das RAG-System.

    Diese Chain verbindet Retrieval und Generation zu einer einfachen Kette.
    """

    def run(self, query: str, **kwargs) -> str:
        """
        Führt die RAG-Chain für eine Query aus.

        Args:
            query: Die Benutzeranfrage
            **kwargs: Weitere Parameter für die Chain

        Returns:
            Die generierte Antwort als String
        """
        # Dokumente abrufen
        print(f"[DEBUG] RAG-Chain: Starte Retrieval für Query: '{query}'")
        documents = self.retriever.retrieve(query)
        print(f"[DEBUG] RAG-Chain: Retrieval abgeschlossen, {len(documents)} Dokumente gefunden")

        # Leerer Kontext als Fallback
        context = "Es stehen keine relevanten Informationen zur Verfügung."

        # Kontext formatieren, falls Dokumente gefunden wurden
        if documents:
            print(f"[DEBUG] RAG-Chain: Formatiere gefundene Dokumente")
            context = self.retriever.format_retrieved_documents(documents)
        else:
            print(f"[DEBUG] RAG-Chain: Keine Dokumente gefunden, verwende Fallback-Kontext")

        # Prompt formatieren
        print(f"[DEBUG] RAG-Chain: Formatiere Prompt für LLM")
        prompt = self.llm.format_prompt(query, context)

        # Antwort generieren
        print(f"[DEBUG] RAG-Chain: Generiere Antwort mit LLM")
        response = self.llm.generate(prompt, **kwargs)
        print(f"[DEBUG] RAG-Chain: Antwort generiert: '{response[:50]}...'")

        return response

    def run_stream(self, query: str, **kwargs) -> Generator[str, None, None]:
        """
        Führt die RAG-Chain für eine Query aus und gibt die Antwort als Stream zurück.

        Args:
            query: Die Benutzeranfrage
            **kwargs: Weitere Parameter für die Chain

        Returns:
            Ein Generator, der die generierte Antwort stückweise zurückgibt
        """
        # Dokumente abrufen
        print(f"[DEBUG] RAG-Chain (Stream): Starte Retrieval für Query: '{query}'")
        documents = self.retriever.retrieve(query)
        print(f"[DEBUG] RAG-Chain (Stream): Retrieval abgeschlossen, {len(documents)} Dokumente gefunden")

        # Leerer Kontext als Fallback
        context = "Es stehen keine relevanten Informationen zur Verfügung."

        # Kontext formatieren, falls Dokumente gefunden wurden
        if documents:
            print(f"[DEBUG] RAG-Chain (Stream): Formatiere gefundene Dokumente")
            context = self.retriever.format_retrieved_documents(documents)
        else:
            print(f"[DEBUG] RAG-Chain (Stream): Keine Dokumente gefunden, verwende Fallback-Kontext")

        # Prompt formatieren
        print(f"[DEBUG] RAG-Chain (Stream): Formatiere Prompt für LLM")
        prompt = self.llm.format_prompt(query, context)

        # Antwort als Stream generieren
        print(f"[DEBUG] RAG-Chain (Stream): Generiere Antwort-Stream mit LLM")
        return self.llm.generate_stream(prompt, **kwargs)


class RerankingRAGChain(RAGChain):
    """
    Implementiert eine RAG-Chain mit Reranking für das RAG-System.

    Diese Chain führt nach dem Retrieval einen zusätzlichen Reranking-Schritt durch,
    um die Relevanz der abgerufenen Dokumente zu verbessern.
    """

    def __init__(self, retriever: Retriever, llm: LLM, reranking_llm: Optional[LLM] = None):
        """
        Initialisiert die RerankingRAGChain.

        Args:
            retriever: Der zu verwendende Retriever
            llm: Das zu verwendende Language Model
            reranking_llm: Optional ein separates Language Model für das Reranking
        """
        super().__init__(retriever, llm)
        self.reranking_llm = reranking_llm or llm

    def _rerank_documents(self, query: str, documents: List[Any]) -> List[Any]:
        """
        Führt ein Reranking der abgerufenen Dokumente durch.

        Args:
            query: Die Benutzeranfrage
            documents: Die abgerufenen Dokumente

        Returns:
            Die neu sortierten Dokumente
        """
        # TODO: Implementiere Reranking-Logik
        # Dies ist ein Platzhalter für zukünftige Erweiterungen
        return documents

    def run(self, query: str, **kwargs) -> str:
        """
        Führt die RerankingRAGChain für eine Query aus.

        Args:
            query: Die Benutzeranfrage
            **kwargs: Weitere Parameter für die Chain

        Returns:
            Die generierte Antwort als String
        """
        # TODO: Implementiere Reranking-Logik
        # Dies ist ein Platzhalter für zukünftige Erweiterungen
        return super().run(query, **kwargs)

    def run_stream(self, query: str, **kwargs) -> Generator[str, None, None]:
        """
        Führt die RerankingRAGChain für eine Query aus und gibt die Antwort als Stream zurück.

        Args:
            query: Die Benutzeranfrage
            **kwargs: Weitere Parameter für die Chain

        Returns:
            Ein Generator, der die generierte Antwort stückweise zurückgibt
        """
        # TODO: Implementiere Reranking-Logik
        # Dies ist ein Platzhalter für zukünftige Erweiterungen
        return super().run_stream(query, **kwargs)