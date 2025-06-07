# src/rag/components/chunking/test_chunker_working.py

import sys
import os

# Pfad hinzufügen (genau wie im Debug-Test)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from src.rag.components.chunking.fixedsize_chunker import FixedSizeChunker
from src.rag.components.data_sources.base import Document

def test_chunking():
    print("🧩 CHUNKING TEST FÜR DSGVO RAG-SYSTEM")
    print("="*50)
    
    # Test-Dokument erstellen
    doc = Document(
        content="DSGVO Artikel 7: Bedingungen für die Einwilligung. " * 20,
        metadata={"article": "7"},
        id="test_doc"
    )
    
    # Chunker testen
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.split_documents([doc])
    
    print(f"Original Länge: {len(doc.content)}")
    print(f"Anzahl Chunks: {len(chunks)}")
    print("-" * 30)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  ID: {chunk.id}")
        print(f"  Länge: {len(chunk.content)}")
        print(f"  Content: {chunk.content[:60]}...")
        print(f"  Metadata: {chunk.metadata}")
        print()
    
    print("✅ Test erfolgreich abgeschlossen!")

if __name__ == "__main__":
    test_chunking()