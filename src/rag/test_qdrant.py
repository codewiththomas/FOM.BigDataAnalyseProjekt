"""
Test-Skript um die Qdrant-Verbindung zu überprüfen.
"""

from src.rag.components.vector_stores.factory import create_vector_store
from src.rag.components.vector_stores.base import Document
import numpy as np

def test_qdrant_connection():
    """Testet die Verbindung zu Qdrant und grundlegende Funktionalität."""
    print("Starte Qdrant-Test...")
    
    # Vector Store erstellen
    vector_store = create_vector_store()
    print(f"Vector Store Typ: {type(vector_store).__name__}")
    
    # Test-Dokument erstellen
    test_doc = Document(
        text="Dies ist ein Test-Dokument",
        embedding=np.random.rand(1536),  # OpenAI Embedding Dimension
        metadata={"source": "test"}
    )
    
    # Dokument hinzufügen
    print("Füge Test-Dokument hinzu...")
    vector_store.add_documents([test_doc])
    
    # Ähnlichkeitssuche durchführen
    print("Führe Ähnlichkeitssuche durch...")
    results = vector_store.similarity_search(
        query_embedding=test_doc.embedding,
        k=1
    )
    
    print(f"Gefundene Dokumente: {len(results)}")
    if results:
        print(f"Erstes Ergebnis: {results[0].text}")

if __name__ == "__main__":
    test_qdrant_connection() 