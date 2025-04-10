from abc import ABC, abstractmethod
from typing import List
from src.rag.components.data_sources.base import Document

class BaseChunker(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Teilt eine Liste von Dokumenten in Chunks.
        """
        pass
