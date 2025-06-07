# src/rag/components/chunking/semantic_chunker.py

import sys
import os

# Konfiguration des Python-Pfads für Modulimports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from typing import List
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from src.rag.components.data_sources.base import Document
from src.rag.components.chunking.base import BaseChunker

import nltk

# NLTK-Punkt-Tokenizer Initialisierung
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SemanticChunker(BaseChunker):
    def __init__(self,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 chunk_size: int = 5,  # Minimale Anzahl Sätze pro Chunk
                 distance_threshold: float = 1.0):
        # Initialisierung des Sentence Transformer Modells
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.distance_threshold = distance_threshold

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Semantische Chunk-Erstellung durch Clustering ähnlicher Sätze.
        
        Args:
            documents: Liste der zu verarbeitenden Dokumente
            
        Returns:
            Liste der erstellten semantischen Chunks als Document-Objekte
        """
        chunked_documents = []

        for doc in documents:
            # Satz-basierte Tokenisierung für semantische Analyse
            sentences = sent_tokenize(doc.content)

            # Verarbeitung kurzer Dokumente ohne Clustering
            if len(sentences) <= self.chunk_size:
                chunked_documents.append(doc)
                continue

            # Generierung von Satz-Embeddings für semantische Ähnlichkeit
            embeddings = self.model.encode(sentences)

            # Anwendung von Agglomerative Clustering für semantische Gruppierung
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric='cosine',
                linkage='average'
            ).fit(embeddings)

            # Gruppierung der Sätze nach Cluster-Labels
            clustered_chunks = {}
            for idx, label in enumerate(clustering.labels_):
                clustered_chunks.setdefault(label, []).append(sentences[idx])

            # Erstellung der Document-Objekte mit Metadaten
            for i, sentence_list in enumerate(clustered_chunks.values()):
                chunk_text = " ".join(sentence_list).strip()
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(clustered_chunks)

                chunked_documents.append(
                    Document(
                        content=chunk_text,
                        metadata=metadata,
                        id=f"{doc.id}_{i}" if doc.id else None
                    )
                )

        return chunked_documents

if __name__ == "__main__":
    try:
        test_doc = Document(
            content="Künstliche Intelligenz revolutioniert die Technologie. Machine Learning ermöglicht neue Anwendungen. Datenschutz ist in Europa wichtig. Die DSGVO regelt den Umgang mit Daten.",
            metadata={}, 
            id="test"
        )
        chunker = SemanticChunker(chunk_size=2, distance_threshold=0.8)
        chunks = chunker.split_documents([test_doc])
        print("SemanticChunker abgeschlossen")
    except Exception as e:
        print(f"Fehler: {e}")