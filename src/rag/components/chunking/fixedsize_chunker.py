# src/rag/components/chunking/fixedsize_chunker.py

import sys
import os

# Konfiguration des Python-Pfads für Modulimports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from typing import List
from src.rag.components.chunking.base import BaseChunker
from src.rag.components.data_sources.base import Document

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1280, chunk_overlap: int = 200):  # Embedding-optimiert
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_documents = []
        
        for doc in documents:
            text = doc.content
            chunks = []
            
            # Implementierung der festen Chunk-Größe mit konfigurierbarem Overlap
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                
                if end >= len(text):
                    break
                start = end - self.chunk_overlap
            
            # Erstellung der Document-Objekte mit Metadaten und Token-Validierung
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.id}_{i}" if doc.id else f"chunk_{i}"
                
                # Token-Validierung mit Warnung
                self.validate_chunk_tokens(chunk, chunk_id)
                
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(chunks)
                metadata["estimated_tokens"] = self.count_tokens_estimate(chunk)

                chunked_documents.append(
                    Document(content=chunk, metadata=metadata, id=chunk_id)
                )
        
        return chunked_documents

if __name__ == "__main__":
    try:
        test_doc = Document(content="Test " * 200, metadata={}, id="test")  # Längerer Test
        chunker = FixedSizeChunker(chunk_size=1280, chunk_overlap=200)
        chunks = chunker.split_documents([test_doc])
        print("FixedSizeChunker abgeschlossen")
    except Exception as e:
        print(f"Fehler: {e}")