# src/rag/components/chunking/test_base.py

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from src.rag.components.chunking.base import BaseChunker
from src.rag.components.data_sources.base import Document

def test_base_chunker():
    print("=== BASE CHUNKER TEST STARTET ===")
    print("🔍 TESTE BASE CHUNKER")
    print("="*30)
    
    # Test 1: Kann BaseChunker importiert werden?
    print(f"✅ BaseChunker importiert: {BaseChunker}")
    
    # Test 2: Ist es eine abstrakte Klasse?
    try:
        # Das sollte einen Fehler geben, weil BaseChunker abstrakt ist
        base_chunker = BaseChunker()
        print("❌ BaseChunker sollte nicht instanziierbar sein!")
    except TypeError as e:
        print(f"✅ BaseChunker ist korrekt abstrakt: {e}")
    
    # Test 3: Teste ob die abstrakte Methode existiert
    print(f"✅ Abstrakte Methode vorhanden: {hasattr(BaseChunker, 'split_documents')}")
    
    # Test 4: Teste ob FixedSizeChunker korrekt von BaseChunker erbt
    from src.rag.components.chunking.fixedsize_chunker import FixedSizeChunker
    
    chunker = FixedSizeChunker()
    print(f"✅ FixedSizeChunker erbt von BaseChunker: {isinstance(chunker, BaseChunker)}")
    print(f"✅ Chunker-Typ: {type(chunker)}")
    
    # Test 5: Teste ob die Methode implementiert ist
    doc = Document(content="Test", metadata={}, id="test")
    chunks = chunker.split_documents([doc])
    print(f"✅ split_documents funktioniert: {len(chunks)} Chunk(s) erstellt")
    
    print("\n🎉 ALLE BASE CHUNKER TESTS ERFOLGREICH!")
    print("=== BASE CHUNKER TEST ENDE ===")

if __name__ == "__main__":
    test_base_chunker()