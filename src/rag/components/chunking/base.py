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
    @staticmethod
    def count_tokens_estimate(text: str) -> int:
        """
        Grobe Token-Schätzung für Embedding-Kompatibilität.
        Berücksichtigt Subword-Tokenization (Faktor 1.3).
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            Geschätzte Anzahl Tokens
        """
        return int(len(text.split()) * 1.3)
    
    @staticmethod
    def validate_chunk_tokens(chunk_content: str, chunk_id: str = "Unknown") -> bool:
        """
        Validierung der Token-Anzahl gegen Embedding-Limits.
        
        Args:
            chunk_content: Inhalt des zu prüfenden Chunks
            chunk_id: ID des Chunks für Warnmeldungen
            
        Returns:
            True wenn unter Limit, False mit Warnung wenn darüber
        """
        token_count = BaseChunker.count_tokens_estimate(chunk_content)
        if token_count > 512:
            print(f"⚠️  WARNUNG: Chunk '{chunk_id}' hat {token_count} Tokens (Limit: 512)")
            return False
        return True

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