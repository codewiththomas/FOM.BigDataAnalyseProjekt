from typing import List, Dict, Any, Optional
from .base_vector_store import BaseVectorStore
import chromadb
from chromadb.config import Settings
import numpy as np


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma Vector Store für persistente und skalierbare Vektorspeicherung.
    """

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents", **kwargs):
        super().__init__(**kwargs)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialisiert den Chroma Client und die Collection.
        """
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )

            # Collection erstellen oder laden
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except:
                self.collection = self.client.create_collection(name=self.collection_name)

        except Exception as e:
            raise Exception(f"Fehler beim Initialisieren von Chroma: {str(e)}")

    def add_documents(self, documents: List[str], embeddings: List[List[float]],
                     metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Fügt Dokumente mit ihren Embeddings zum Vector Store hinzu.
        """
        if not documents or not embeddings:
            return

        # IDs für die Dokumente generieren
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Metadata vorbereiten
        if metadata is None:
            metadata = [{"source": f"document_{i}"} for i in range(len(documents))]

        try:
            # Dokumente zur Collection hinzufügen
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )

        except Exception as e:
            raise Exception(f"Fehler beim Hinzufügen von Dokumenten: {str(e)}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sucht nach ähnlichen Dokumenten basierend auf einem Query-Embedding.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Ergebnisse formatieren
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })

            return formatted_results

        except Exception as e:
            raise Exception(f"Fehler bei der Suche: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über die Collection zurück.
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}

    def clear(self) -> None:
        """
        Löscht alle Dokumente aus der Collection.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
        except Exception as e:
            raise Exception(f"Fehler beim Löschen der Collection: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        return {
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name
        }