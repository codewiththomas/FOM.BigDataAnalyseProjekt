from typing import List
from .base_chunker import BaseChunker


class LineChunker(BaseChunker):
    """
    Chunker, der Text nach Zeilen aufteilt.

    Respektiert dabei die Chunk-Größe und Überlappung.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50, **kwargs):
        """
        Initialisiert den LineChunker.

        Args:
            chunk_size: Maximale Größe eines Chunks in Zeichen
            overlap: Überlappung zwischen Chunks in Zeichen
            **kwargs: Weitere Parameter
        """
        super().__init__(chunk_size, overlap, **kwargs)

    def chunk_text(self, text: str) -> List[str]:
        """
        Teilt Text in Chunks basierend auf Zeilen auf.

        Args:
            text: Der zu teilende Text

        Returns:
            Liste von Text-Chunks
        """
        if not text.strip():
            return []

        # Text in Zeilen aufteilen
        lines = text.split('\n')
        chunks = []
        current_chunk = ""

        for line in lines:
            # Prüfen, ob die Zeile zu lang ist
            if len(line) > self.chunk_size:
                # Wenn aktueller Chunk nicht leer ist, speichern
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Lange Zeile in kleinere Teile aufteilen
                chunks.extend(self._split_long_line(line))
                continue

            # Prüfen, ob die Zeile in den aktuellen Chunk passt
            potential_chunk = current_chunk + "\n" + line if current_chunk else line

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Aktuellen Chunk speichern
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Neuen Chunk mit Überlappung starten
                current_chunk = self._create_overlapping_chunk(current_chunk, line)

        # Letzten Chunk speichern
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_line(self, line: str) -> List[str]:
        """
        Teilt eine zu lange Zeile in kleinere Chunks auf.

        Args:
            line: Die zu teilende Zeile

        Returns:
            Liste von Chunks
        """
        chunks = []
        start = 0

        while start < len(line):
            end = start + self.chunk_size
            chunk = line[start:end]
            chunks.append(chunk)
            start = end - self.overlap

        return chunks

    def _create_overlapping_chunk(self, previous_chunk: str, new_line: str) -> str:
        """
        Erstellt einen neuen Chunk mit Überlappung zum vorherigen.

        Args:
            previous_chunk: Der vorherige Chunk
            new_line: Die neue Zeile

        Returns:
            Neuer Chunk mit Überlappung
        """
        if self.overlap <= 0:
            return new_line

        # Die letzten Zeichen des vorherigen Chunks als Überlappung nehmen
        overlap_text = previous_chunk[-self.overlap:] if len(previous_chunk) > self.overlap else previous_chunk

        # Sicherstellen, dass die Überlappung nicht zu lang wird
        if len(overlap_text) + len(new_line) + 1 <= self.chunk_size:
            return overlap_text + "\n" + new_line
        else:
            return new_line