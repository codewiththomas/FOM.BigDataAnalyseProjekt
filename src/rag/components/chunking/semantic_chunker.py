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
                 chunk_size: int = 3,  # Embedding-optimiert: weniger Sätze pro Chunk
                 distance_threshold: float = 0.8):  # Embedding-optimiert: engere semantische Gruppierung
        # Initialisierung des Sentence Transformer Modells
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.distance_threshold = distance_threshold

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Semantische Chunk-Erstellung durch Clustering ähnlicher Sätze
        mit Token-Validierung für Embedding-Kompatibilität.
        
        Args:
            documents: Liste der zu verarbeitenden Dokumente
            
        Returns:
            Liste der erstellten semantischen Chunks als Document-Objekte mit Token-Metadaten
        """
        chunked_documents = []

        for doc in documents:
            # Satz-basierte Tokenisierung für semantische Analyse
            sentences = sent_tokenize(doc.content)

            # Verarbeitung kurzer Dokumente ohne Clustering
            if len(sentences) <= self.chunk_size:
                # Token-Validierung auch für ungeclusterte Dokumente
                chunk_id = doc.id or "short_doc"
                self.validate_chunk_tokens(doc.content, chunk_id)
                
                # Erweiterung der Metadaten um Token-Information
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["estimated_tokens"] = self.count_tokens_estimate(doc.content)
                
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

            # Erstellung der Document-Objekte mit Metadaten und Token-Validierung
            for i, sentence_list in enumerate(clustered_chunks.values()):
                chunk_text = " ".join(sentence_list).strip()
                chunk_id = f"{doc.id}_{i}" if doc.id else f"chunk_{i}"
                
                # Token-Validierung mit Warnung
                self.validate_chunk_tokens(chunk_text, chunk_id)
                
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(clustered_chunks)
                metadata["estimated_tokens"] = self.count_tokens_estimate(chunk_text)

                chunked_documents.append(
                    Document(
                        content=chunk_text,
                        metadata=metadata,
                        id=chunk_id
                    )
                )

        return chunked_documents

if __name__ == "__main__":
    try:
        test_doc = Document(
            content="Künstliche Intelligenz revolutioniert die Technologie. Machine Learning ermöglicht neue Anwendungen. Datenschutz ist in Europa wichtig. Die DSGVO regelt den Umgang mit Daten. Deep Learning verwendet neuronale Netze. Algorithmen verarbeiten große Datenmengen. Automatisierung verändert Arbeitsprozesse. Ethik in der KI wird diskutiert." * 5,  # Längerer Test
            metadata={}, 
            id="test"
        )
        chunker = SemanticChunker(chunk_size=3, distance_threshold=0.8)
        chunks = chunker.split_documents([test_doc])
        print("SemanticChunker abgeschlossen")
    except Exception as e:
        print(f"Fehler: {e}")