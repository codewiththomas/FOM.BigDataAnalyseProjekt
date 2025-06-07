# src/rag/components/chunking/nltk_chunker.py

import sys
import os

# Pfad hinzufügen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from src.rag.components.data_sources.base import Document
from src.rag.components.chunking.base import BaseChunker

# NLTK-Punkt-Tokenizer initialisierung
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class NLTKChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_documents = []

        for doc in documents:
            # Satz-basierte Tokenisierung mit NLTK
            sentences = sent_tokenize(doc.content)
            chunks = []
            current_chunk = ""
            
            # Zusammenfügen von Sätzen bis zur maximalen Chunk-Größe
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk += (" " + sentence) if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Implementierung von Overlap-Mechanismus auf Satzebene
            if self.chunk_overlap > 0 and len(chunks) > 1:
                new_chunks = [chunks[0]]
                for i in range(1, len(chunks)):
                    previous_sentences = sent_tokenize(chunks[i - 1])
                    current_chunk = chunks[i]
                    
                    # Verwendung der letzten Sätze als Overlap-Kontext
                    overlap_sentences = previous_sentences[-2:] if len(previous_sentences) >= 2 else previous_sentences
                    overlap = " ".join(overlap_sentences)
                    
                    # Begrenzung des Overlaps auf konfigurierte Zeichenanzahl
                    if len(overlap) > self.chunk_overlap:
                        overlap = overlap[-self.chunk_overlap:]
                    
                    new_chunks.append(overlap + " " + current_chunk)
                chunks = new_chunks

            # Erstellung der Document-Objekte mit Metadaten
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(chunks)

                chunked_documents.append(
                    Document(content=chunk, metadata=metadata, id=f"{doc.id}_{i}" if doc.id else None)
                )

        return chunked_documents

if __name__ == "__main__":
    try:
        test_doc = Document(content="Das ist Satz eins. Das ist Satz zwei. Das ist Satz drei." * 10, metadata={}, id="test")
        chunker = NLTKChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_documents([test_doc])
        print("NLTKChunker abgeschlossen")
    except Exception as e:
        print(f"Fehler: {e}")