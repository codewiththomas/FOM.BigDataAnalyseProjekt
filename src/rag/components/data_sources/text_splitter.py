"""
Implementiert Text-Splitter für verschiedene Dokument-Typen.
"""

import re
from typing import List, Dict, Any, Optional, Callable

from .base import Document, TextSplitter


class CharacterTextSplitter(TextSplitter):
    """
    Teilt Text anhand von Zeichen auf.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialisiert den CharacterTextSplitter.

        Args:
            chunk_size: Maximale Größe eines Chunks in Zeichen
            chunk_overlap: Überlappung zwischen Chunks in Zeichen
            separator: Trennzeichen für die Aufteilung des Textes
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """
        Teilt einen Text in Chunks auf.

        Args:
            text: Der aufzuteilende Text

        Returns:
            Eine Liste von Text-Chunks
        """
        if not text.strip():
            return []

        # Text in kleinere Teile aufteilen
        splits = text.split(self.separator)
        splits = [s for s in splits if s.strip()]

        # Chunks erstellen
        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split)

            if current_size + split_size + len(self.separator) <= self.chunk_size:
                # Split passt noch in den aktuellen Chunk
                current_chunk.append(split)
                current_size += split_size + len(self.separator)
            else:
                # Aktuellen Chunk abschließen und neuen starten
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))

                # Neuen Chunk starten
                current_chunk = [split]
                current_size = split_size

        # Letzten Chunk hinzufügen, falls vorhanden
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        # Überlappende Chunks erstellen
        if self.chunk_overlap > 0 and len(chunks) > 1:
            new_chunks = [chunks[0]]

            for i in range(1, len(chunks)):
                current_text = chunks[i]
                previous_text = chunks[i-1]

                # Überlappung erstellen
                if len(previous_text) > self.chunk_overlap:
                    overlap_text = previous_text[-self.chunk_overlap:]
                    current_with_overlap = overlap_text + self.separator + current_text
                    new_chunks.append(current_with_overlap)
                else:
                    new_chunks.append(current_text)

            chunks = new_chunks

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Teilt Dokumente in kleinere Chunks auf.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Document-Objekten, wobei jedes Dokument einen Chunk darstellt
        """
        chunked_documents = []

        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(chunks)

                chunked_documents.append(
                    Document(content=chunk, metadata=metadata, id=f"{doc.id}_{i}" if doc.id else None)
                )

        return chunked_documents


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Teilt Text rekursiv mit verschiedenen Trennzeichen auf.

    Dies ist nützlich für strukturierten Text wie Gesetze.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    ):
        """
        Initialisiert den RecursiveCharacterTextSplitter.

        Args:
            chunk_size: Maximale Größe eines Chunks in Zeichen
            chunk_overlap: Überlappung zwischen Chunks in Zeichen
            separators: Liste von Trennzeichen in absteigender Priorität
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        Teilt einen Text rekursiv in Chunks auf.

        Args:
            text: Der aufzuteilende Text

        Returns:
            Eine Liste von Text-Chunks
        """
        # Rekursive Funktion zum Aufteilen des Textes
        def _split_text_recursive(text: str, separators: List[str], depth: int = 0) -> List[str]:
            # Wenn wir beim letzten Separator angelangt sind oder der Text klein genug ist
            if depth == len(separators) - 1 or len(text) <= self.chunk_size:
                separator = separators[depth]
                splits = text.split(separator)

                # Leere Splits entfernen
                splits = [s for s in splits if s.strip()]

                # Chunks erstellen
                chunks = []
                current_chunk = []
                current_size = 0

                for split in splits:
                    split_size = len(split)

                    if current_size + split_size + len(separator) <= self.chunk_size:
                        # Split passt noch in den aktuellen Chunk
                        current_chunk.append(split)
                        current_size += split_size + len(separator)
                    else:
                        # Aktuellen Chunk abschließen und neuen starten
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))

                        # Falls der Split selbst zu groß ist und wir nicht beim letzten Separator sind
                        if split_size > self.chunk_size and depth < len(separators) - 1:
                            # Rekursiv mit dem nächsten Separator aufteilen
                            sub_chunks = _split_text_recursive(split, separators, depth + 1)
                            chunks.extend(sub_chunks)
                            current_chunk = []
                            current_size = 0
                        else:
                            # Neuen Chunk starten
                            current_chunk = [split]
                            current_size = split_size

                # Letzten Chunk hinzufügen, falls vorhanden
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                return chunks
            else:
                # Text mit aktuellem Separator aufteilen
                separator = separators[depth]
                splits = text.split(separator)

                # Rekursiv jeden Split mit dem nächsten Separator aufteilen
                chunks = []
                for split in splits:
                    if len(split.strip()) > 0:
                        sub_chunks = _split_text_recursive(split, separators, depth + 1)
                        chunks.extend(sub_chunks)

                return chunks

        chunks = _split_text_recursive(text, self.separators)

        # Überlappung hinzufügen
        if self.chunk_overlap > 0 and len(chunks) > 1:
            new_chunks = [chunks[0]]

            for i in range(1, len(chunks)):
                current_text = chunks[i]
                previous_text = chunks[i-1]

                # Überlappung erstellen
                if len(previous_text) > self.chunk_overlap:
                    overlap_text = previous_text[-self.chunk_overlap:]
                    current_with_overlap = overlap_text + chunks[i]
                    new_chunks.append(current_with_overlap)
                else:
                    new_chunks.append(current_text)

            chunks = new_chunks

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Teilt Dokumente in kleinere Chunks auf.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Document-Objekten, wobei jedes Dokument einen Chunk darstellt
        """
        chunked_documents = []

        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk"] = i
                metadata["chunk_count"] = len(chunks)

                chunked_documents.append(
                    Document(content=chunk, metadata=metadata, id=f"{doc.id}_{i}" if doc.id else None)
                )

        return chunked_documents