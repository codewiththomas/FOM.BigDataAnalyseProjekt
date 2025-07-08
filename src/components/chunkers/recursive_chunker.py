from typing import List, Dict, Any
from .base_chunker import BaseChunker
import re


class RecursiveChunker(BaseChunker):
    """
    Recursive Character Text Splitter - Baseline Implementation

    Teilt Text rekursiv an verschiedenen Trennzeichen auf:
    1. Absätze (\n\n)
    2. Zeilen (\n)
    3. Sätze (. ! ?)
    4. Wörter ( )
    5. Zeichen
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialisiert den Recursive Chunker.

        Args:
            chunk_size: Maximale Chunk-Größe in Zeichen
            chunk_overlap: Überlappung zwischen Chunks in Zeichen
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarchie der Trenner (von grob zu fein)
        self.separators = [
            "\n\n",  # Absätze
            "\n",    # Zeilen
            ". ",    # Sätze
            "! ",    # Ausrufe
            "? ",    # Fragen
            " ",     # Wörter
            ""       # Zeichen
        ]

    def split_document(self, document: str, chunk_size: int = 1000) -> List[str]:
        """
        Teilt ein Dokument rekursiv in Chunks auf.

        Args:
            document: Zu teilendes Dokument
            chunk_size: Optionale Chunk-Größe (überschreibt Standard)

        Returns:
            Liste von Chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        if not document or not document.strip():
            return []

        return self._split_text(document, chunk_size)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Rekursive Textaufteilung.

        Args:
            text: Zu teilender Text
            chunk_size: Maximale Chunk-Größe

        Returns:
            Liste von Chunks
        """
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        # Versuche jeden Separator in der Hierarchie
        for separator in self.separators:
            if separator in text:
                return self._split_by_separator(text, separator, chunk_size)

        # Fallback: Erzwinge Aufteilung nach Zeichen
        return self._force_split(text, chunk_size)

    def _split_by_separator(self, text: str, separator: str, chunk_size: int) -> List[str]:
        """
        Teilt Text an einem bestimmten Separator auf.

        Args:
            text: Zu teilender Text
            separator: Trennzeichen
            chunk_size: Maximale Chunk-Größe

        Returns:
            Liste von Chunks
        """
        splits = text.split(separator)

        # Separator wieder hinzufügen (außer bei leerem Separator)
        if separator:
            splits = [split + separator for split in splits[:-1]] + [splits[-1]]

        return self._merge_splits(splits, chunk_size)

    def _merge_splits(self, splits: List[str], chunk_size: int) -> List[str]:
        """
        Fügt kleine Splits zu größeren Chunks zusammen.

        Args:
            splits: Liste von Text-Splits
            chunk_size: Maximale Chunk-Größe

        Returns:
            Liste von Chunks
        """
        chunks = []
        current_chunk = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            # Wenn Split zu groß ist, rekursiv aufteilen
            if len(split) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Rekursiv aufteilen
                sub_chunks = self._split_text(split, chunk_size)
                chunks.extend(sub_chunks)
                continue

            # Prüfe ob Split in aktuellen Chunk passt
            if len(current_chunk) + len(split) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + split
                else:
                    current_chunk = split
            else:
                # Aktueller Chunk ist voll, starte neuen
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = split

        # Letzten Chunk hinzufügen
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Overlap hinzufügen
        return self._add_overlap(chunks)

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Fügt Überlappung zwischen Chunks hinzu.

        Args:
            chunks: Liste von Chunks ohne Überlappung

        Returns:
            Liste von Chunks mit Überlappung
        """
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            # Overlap vom vorherigen Chunk
            prev_chunk = chunks[i-1]
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # Overlap am Anfang des aktuellen Chunks hinzufügen
            overlapped_chunk = overlap_text + " " + chunk
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _force_split(self, text: str, chunk_size: int) -> List[str]:
        """
        Erzwingt Aufteilung nach Zeichen wenn keine Separator funktionieren.

        Args:
            text: Zu teilender Text
            chunk_size: Maximale Chunk-Größe

        Returns:
            Liste von Chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Versuche an Wortgrenze zu trennen
            if end < len(text):
                # Suche rückwärts nach Leerzeichen
                while end > start and text[end] != ' ':
                    end -= 1

                # Wenn kein Leerzeichen gefunden, erzwinge Trennung
                if end == start:
                    end = start + chunk_size

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def get_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators
        }