from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):

    @abstractmethod
    def split_document(self, document: str, chunk_size: int) -> List[str]:
        """
        Splitted ein Dokument in kleinere Chunks.

        Args:
            document (str): Der Inhalt des Dokuments, das in Chunks aufgeteilt werden soll.
            chunk_size (int): Die maximale Größe jedes Chunks.

        Returns:
            list[str]: A list of document chunks.
        """
        pass