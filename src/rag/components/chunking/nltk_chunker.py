from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from src.rag.components.data_sources.base import Document
from src.rag.components.chunking.base import BaseChunker

nltk.download('punkt')

class NLTKChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_documents = []

        for doc in documents:
            sentences = sent_tokenize(doc.content)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Overlap hinzufügen (optional, einfaches Anhängen der letzten n Zeichen)
            if self.chunk_overlap > 0 and len(chunks) > 1:
                new_chunks = [chunks[0]]
                for i in range(1, len(chunks)):
                    previous = chunks[i - 1]
                    current = chunks[i]
                    overlap = previous[-self.chunk_overlap:] if len(previous) > self.chunk_overlap else previous
                    new_chunks.append(overlap + " " + current)
                chunks = new_chunks

            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(chunks)

                chunked_documents.append(
                    Document(content=chunk, metadata=metadata, id=f"{doc.id}_{i}" if doc.id else None)
                )

        return chunked_documents
