"""
Basisklassen und Interfaces für Datenquellen im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Document:
    """
    Repräsentiert ein Dokument im System.

    Attributes:
        content: Der Textinhalt des Dokuments
        metadata: Zusätzliche Metadaten zum Dokument (Quelle, Datum, Autor, etc.)
        id: Optional eindeutige ID des Dokuments
    """
    content: str
    metadata: Dict[str, Any] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentLoader(ABC):
    """
    Basisklasse für alle Dokumentenlader.

    Ermöglicht das Laden von Dokumenten aus verschiedenen Quellen.
    """

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Lädt Dokumente aus der Quelle.

        Returns:
            Eine Liste von Document-Objekten
        """
        pass


class TextSplitter(ABC):
    """
    Basisklasse für alle Text-Splitter.

    Ermöglicht das Aufteilen von Dokumenten in kleinere Chunks.
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Teilt Dokumente in kleinere Chunks auf.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Document-Objekten, wobei jedes Dokument einen Chunk darstellt
        """
        pass