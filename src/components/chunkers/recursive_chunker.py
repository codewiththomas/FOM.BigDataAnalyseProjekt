from typing import List, Dict, Any
from .base_chunker import BaseChunker
import re


class RecursiveCharacterChunker(BaseChunker):
    """
    Erweiterter Recursive Character Chunker für bessere Textaufteilung.
    Teilt Text basierend auf verschiedenen Trennzeichen in der Reihenfolge ihrer Priorität.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Trennzeichen in der Reihenfolge ihrer Priorität
        self.separators = [
            "\n\n",  # Doppelte Zeilenumbrüche (Absätze)
            "\n",    # Einzelne Zeilenumbrüche
            ". ",    # Sätze
            "! ",    # Ausrufezeichen
            "? ",    # Fragezeichen
            "; ",    # Semikolon
            ": ",    # Doppelpunkt
            ", ",    # Komma
            " ",     # Leerzeichen
            ""       # Fallback: Zeichen für Zeichen
        ]

    def chunk_text(self, text: str) -> List[str]:
        """
        Teilt Text rekursiv basierend auf den definierten Trennzeichen.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Text in Sätze aufteilen
        sentences = self._split_text(text)

        for sentence in sentences:
            # Wenn der aktuelle Chunk + neuer Satz zu lang wird
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: Letzte Sätze des vorherigen Chunks beibehalten
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence

        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Teilt Text basierend auf den definierten Trennzeichen.
        """
        # Bereinige Text
        text = re.sub(r'\s+', ' ', text).strip()

        # Verwende das erste Trennzeichen, das funktioniert
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                # Füge Trennzeichen wieder hinzu (außer beim letzten Teil)
                result = []
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        result.append(part + separator)
                    else:
                        result.append(part)
                return [part.strip() for part in result if part.strip()]

        # Fallback: Zeichen für Zeichen
        return [text]

    def _get_overlap_text(self, text: str) -> str:
        """
        Extrahiert den Overlap-Text vom Ende des Chunks.
        """
        if self.chunk_overlap <= 0:
            return ""

        # Finde die letzten Sätze, die in den Overlap passen
        sentences = self._split_text(text)
        overlap_text = ""

        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= self.chunk_overlap:
                overlap_text = sentence + overlap_text
            else:
                break

        return overlap_text

    def get_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators
        }