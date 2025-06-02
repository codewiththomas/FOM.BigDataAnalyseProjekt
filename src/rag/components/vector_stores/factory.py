"""
Factory für Vector Stores.
"""

from qdrant_client import QdrantClient
from src.rag.components.vector_stores.base import BaseVectorStore
from src.rag.components.vector_stores.qdrant_store import QdrantVectorStore
from src.rag.config import config


def create_vector_store() -> BaseVectorStore:
    """
    Erstellt einen Vector Store basierend auf der Konfiguration.
    
    Returns:
        Eine Instanz eines Vector Stores
    """
    # Erstelle Qdrant Client mit Konfigurationswerten
    client = QdrantClient(
        host=config.vector_store.qdrant_host,
        port=config.vector_store.qdrant_port
    )
    
    # Erstelle und gebe den QdrantVectorStore zurück
    return QdrantVectorStore(
        client=client,
        collection_name=config.vector_store.collection_name
    ) 