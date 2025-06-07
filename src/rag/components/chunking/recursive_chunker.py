# src/rag/components/chunking/recursive_chunker.py

import sys
import os

# Konfiguration des Python-Pfads für Modulimports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from typing import List
from src.rag.components.data_sources.base import Document
from src.rag.components.chunking.base import BaseChunker
from src.rag.components.data_sources.text_splitter import RecursiveCharacterTextSplitter

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators=None):
        # Definition der hierarchischen Trennzeichen für rekursive Aufteilung
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        # Initialisierung des rekursiven Text-Splitters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Verwendung des rekursiven Splitters zur intelligenten Dokumentaufteilung.
        
        Args:
            documents: Liste der zu verarbeitenden Dokumente
            
        Returns:
            Liste der erstellten Chunks als Document-Objekte
        """
        return self.splitter.split_documents(documents)

if __name__ == "__main__":
    try:
        test_doc = Document(content="Absatz eins.\n\nAbsatz zwei.\nZeile drei. Satz vier.", metadata={}, id="test")
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split_documents([test_doc])
        print("RecursiveChunker abgeschlossen")
    except Exception as e:
        print(f"Fehler: {e}")