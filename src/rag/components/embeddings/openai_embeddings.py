"""
Implementiert OpenAI-basierte Embeddings für das RAG-System.
"""

import os
from typing import List
import numpy as np
from openai import OpenAI

from ...config import config
from ...components.data_sources.base import Document as SourceDocument
from ...components.vector_stores.base import Document as VectorDocument
from .base import Embedder


class OpenAIEmbedder(Embedder):
    """
    Implementiert Embeddings mit OpenAI.
    """

    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialisiert den OpenAIEmbedder.

        Args:
            model: Das zu verwendende Embedding-Modell
            api_key: Der OpenAI API-Key
        """
        self.model = model or config.embedding.model
        self.api_key = api_key or config.embedding.api_key
        self.client = OpenAI(api_key=self.api_key)

    def _get_document_texts(self, documents: List[SourceDocument]) -> List[str]:
        """
        Extrahiert die Texte aus den Dokumenten.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Texten
        """
        return [doc.content for doc in documents]

    def embed_query(self, text: str) -> np.ndarray:
        """
        Erstellt ein Embedding für eine einzelne Query mit OpenAI.

        Args:
            text: Der zu embedende Text

        Returns:
            Ein Embedding-Vektor als numpy Array
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_documents(self, documents: List[SourceDocument]) -> List[VectorDocument]:
        """
        Erstellt Embeddings für eine Liste von Dokumenten mit OpenAI.

        Args:
            documents: Eine Liste von Document-Objekten aus data_sources

        Returns:
            Eine Liste von Document-Objekten für den Vector Store
        """
        if not documents:
            return []

        texts = self._get_document_texts(documents)
        vector_documents = []

        # OpenAI hat ein Limit für die Anzahl der Tokens, die in einem API-Call
        # verarbeitet werden können. Daher teilen wir die Dokumente in Batches auf.
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                batch_embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
                
                # Erstelle Vector Store Documents
                for doc, embedding in zip(batch_docs, batch_embeddings):
                    # Stelle sicher, dass die Metadaten kopiert werden
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    
                    # Erstelle ein neues VectorDocument
                    vector_doc = VectorDocument(
                        text=doc.content,  # Verwende content von SourceDocument als text
                        embedding=embedding,
                        metadata=metadata
                    )
                    vector_documents.append(vector_doc)
                    
            except Exception as e:
                print(f"Fehler beim Embedden von Batch {i}-{i+batch_size}: {e}")
                print("Versuche, Dokumente einzeln zu embedden...")

                for j, doc in enumerate(batch_docs):
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=texts[i+j]
                        )
                        embedding = np.array(response.data[0].embedding, dtype=np.float32)
                        
                        # Stelle sicher, dass die Metadaten kopiert werden
                        metadata = doc.metadata.copy() if doc.metadata else {}
                        
                        # Erstelle ein neues VectorDocument
                        vector_doc = VectorDocument(
                            text=doc.content,  # Verwende content von SourceDocument als text
                            embedding=embedding,
                            metadata=metadata
                        )
                        vector_documents.append(vector_doc)
                    except Exception as e:
                        print(f"Fehler beim Embedden von Dokument {i+j}: {e}")
                        # Dummy-Embedding erstellen, um die Reihenfolge beizubehalten
                        dummy_embedding = np.zeros(1536, dtype=np.float32)  # OpenAI Embeddings haben 1536 Dimensionen
                        
                        # Stelle sicher, dass die Metadaten kopiert werden
                        metadata = doc.metadata.copy() if doc.metadata else {}
                        
                        # Erstelle ein neues VectorDocument
                        vector_doc = VectorDocument(
                            text=doc.content,  # Verwende content von SourceDocument als text
                            embedding=dummy_embedding,
                            metadata=metadata
                        )
                        vector_documents.append(vector_doc)

        return vector_documents