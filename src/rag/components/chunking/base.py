# src/rag/components/chunking/base.py

import sys
import os

# Konfiguration des Python-Pfads für Modulimports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import List
from src.rag.components.data_sources.base import Document

class BaseChunker(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Abstrakte Methode zur Aufteilung von Dokumenten in Chunks.
        
        Args:
            documents: Liste der zu verarbeitenden Dokumente
            
        Returns:
            Liste der erstellten Chunks als Document-Objekte
        """
        pass

# Bestätigung des erfolgreichen Modulimports
print(f"📦 BaseChunker Modul geladen")