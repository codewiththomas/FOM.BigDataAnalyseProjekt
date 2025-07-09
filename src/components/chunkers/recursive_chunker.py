import re
from typing import List, Optional
from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Recursive text chunker that splits text using a hierarchy of separators.

    This chunker attempts to split text at natural boundaries (paragraphs, sentences, words)
    while respecting the maximum chunk size. It uses a recursive approach, trying
    different separators in order of preference.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None,
                 keep_separator: bool = True):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try in order (default: paragraph, sentence, word)
            keep_separator: Whether to keep the separator in the chunks
        """
        super().__init__(chunk_size, chunk_overlap)

        if separators is None:
            # Default separators in order of preference
            self.separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Semicolon
                ", ",    # Comma
                " ",     # Spaces
                ""       # Character level (last resort)
            ]
        else:
            self.separators = separators

        self.keep_separator = keep_separator

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive approach.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Clean the text
        text = text.strip()

        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Start recursive chunking
        chunks = self._split_text_recursive(text, self.separators)

        # Apply overlap if needed
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the provided separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        # Base case: if text is small enough, return it
        if len(text) <= self.chunk_size:
            return [text]

        # Base case: if no more separators, split by character
        if not separators:
            return self._split_by_character(text)

        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level splitting
            return self._split_by_character(text)

        # Split by current separator
        splits = self._split_by_separator(text, separator)

        # If we couldn't split or only got one piece, try next separator
        if len(splits) <= 1:
            return self._split_text_recursive(text, remaining_separators)

        # Process each split
        chunks = []
        current_chunk = ""

        for split in splits:
            # If adding this split would exceed chunk size
            if len(current_chunk) + len(split) > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If split itself is too large, recursively split it
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(split, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = split
            else:
                # Add to current chunk
                current_chunk += split

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text by a specific separator.

        Args:
            text: Text to split
            separator: Separator to use

        Returns:
            List of text pieces
        """
        if separator in text:
            splits = text.split(separator)

            if self.keep_separator and separator != " ":
                # Add separator back to all pieces except the last
                result = []
                for i, split in enumerate(splits):
                    if i < len(splits) - 1:
                        result.append(split + separator)
                    else:
                        result.append(split)
                return result
            else:
                return splits
        else:
            return [text]

    def _split_by_character(self, text: str) -> List[str]:
        """
        Split text by character when no other separator works.

        Args:
            text: Text to split

        Returns:
            List of character-level chunks
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.

        Args:
            chunks: List of chunks without overlap

        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i-1]

            # Get overlap from previous chunk
            if len(previous_chunk) > self.chunk_overlap:
                overlap = previous_chunk[-self.chunk_overlap:]
                current_chunk = overlap + current_chunk

            overlapped_chunks.append(current_chunk)

        return overlapped_chunks

    def get_chunk_metadata(self, chunk: str, chunk_index: int) -> dict:
        """
        Get metadata for a specific chunk.

        Args:
            chunk: The chunk text
            chunk_index: Index of the chunk

        Returns:
            Dictionary containing chunk metadata
        """
        metadata = super().get_chunk_metadata(chunk, chunk_index)
        metadata.update({
            "chunker_type": "recursive",
            "separators_used": self.separators,
            "keep_separator": self.keep_separator,
            "estimated_sentences": len(re.findall(r'[.!?]+', chunk)),
            "estimated_words": len(chunk.split()),
            "has_paragraph_breaks": "\n\n" in chunk,
            "has_line_breaks": "\n" in chunk
        })
        return metadata