# src/rag/components/chunking/debug_test.py

import sys
import os

# Pfad hinzufügen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

print("=== DEBUG TEST ===")
print(f"Aktueller Ordner: {os.path.dirname(__file__)}")
print(f"Projekt Root: {project_root}")

# Test 1: Kann Document importiert werden?
try:
    from src.rag.components.data_sources.base import Document
    print("✅ Document Import erfolgreich")
except Exception as e:
    print(f"❌ Document Import Fehler: {e}")

# Test 2: Kann BaseChunker importiert werden?
try:
    from src.rag.components.chunking.base import BaseChunker
    print("✅ BaseChunker Import erfolgreich")
except Exception as e:
    print(f"❌ BaseChunker Import Fehler: {e}")

# Test 3: Kann FixedSizeChunker importiert werden?
try:
    from src.rag.components.chunking.fixedsize_chunker import FixedSizeChunker
    print("✅ FixedSizeChunker Import erfolgreich")
    
    # Test 4: Kann eine Instanz erstellt werden?
    chunker = FixedSizeChunker()
    print("✅ FixedSizeChunker Instanz erstellt")
    
except Exception as e:
    print(f"❌ FixedSizeChunker Fehler: {e}")

print("=== DEBUG ENDE ===")