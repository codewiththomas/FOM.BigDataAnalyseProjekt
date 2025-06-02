"""
Basisklassen und Interfaces für RAG-Chains im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Generator

from ...components.retrieval.base import Retriever
from ...components.llm.base import LLM


class RAGChain(ABC):
    """
    Basisklasse für alle RAG-Chains.

    Verbindet Retrieval und Generation zu einer Kette.
    """

    def __init__(self, retriever: Retriever, llm: LLM):
        """
        Initialisiert die RAG-Chain.

        Args:
            retriever: Der zu verwendende Retriever
            llm: Das zu verwendende Language Model
        """
        self.retriever = retriever
        self.llm = llm

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """
        Führt die RAG-Chain für eine Query aus.

        Args:
            query: Die Benutzeranfrage
            **kwargs: Weitere Parameter für die Chain

        Returns:
            Die generierte Antwort als String
        """
        pass

    @abstractmethod
    def run_stream(self, query: str, **kwargs) -> Generator[str, None, None]:
        """
        Führt die RAG-Chain für eine Query aus und gibt die Antwort als Stream zurück.

        Args:
            query: Die Benutzeranfrage
            **kwargs: Weitere Parameter für die Chain

        Returns:
            Ein Generator, der die generierte Antwort stückweise zurückgibt
        """
        pass