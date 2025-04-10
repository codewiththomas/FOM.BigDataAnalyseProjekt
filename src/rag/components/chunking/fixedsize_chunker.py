from typing import List
from src.rag.components.chunking.base import BaseChunker
from src.rag.components.data_sources.base import Document

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_documents = []

        for doc in documents:
            text = doc.content
            chunks = [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(chunks)

                chunked_documents.append(
                    Document(content=chunk, metadata=metadata, id=f"{doc.id}_{i}" if doc.id else None)
                )

        return chunked_documents
