import re
import numpy as np
from typing import List, Optional, Tuple
from .base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic text chunker that splits text based on semantic similarity.

    This chunker uses sentence embeddings to determine semantic boundaries
    and groups semantically similar sentences together while respecting
    the maximum chunk size.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 similarity_threshold: float = 0.7,
                 min_sentences_per_chunk: int = 2,
                 embedding_model: Optional[str] = None):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            similarity_threshold: Threshold for semantic similarity (0-1)
            min_sentences_per_chunk: Minimum number of sentences per chunk
            embedding_model: Name of the embedding model to use
        """
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"

        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model)
            self.use_embeddings = True
        except ImportError:
            print("Warning: sentence-transformers not available. Using fallback method.")
            self.encoder = None
            self.use_embeddings = False

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic similarity.

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

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= self.min_sentences_per_chunk:
            return [text]

        # Create semantic chunks
        if self.use_embeddings:
            chunks = self._create_semantic_chunks(sentences)
        else:
            # Fallback to simple sentence-based chunking
            chunks = self._create_sentence_chunks(sentences)

        # Apply overlap if needed
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_semantic_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create chunks based on semantic similarity using embeddings.

        Args:
            sentences: List of sentences

        Returns:
            List of semantic chunks
        """
        if not sentences:
            return []

        # Get sentence embeddings
        embeddings = self.encoder.encode(sentences)

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(embeddings)

        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(similarity_matrix)

        # Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(sentences, boundaries)

        return chunks

    def _create_sentence_chunks(self, sentences: List[str]) -> List[str]:
        """
        Fallback method to create chunks based on sentence count and size.

        Args:
            sentences: List of sentences

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                # Save current chunk if it has minimum sentences
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # If single sentence is too long, split it
                    if len(sentence) > self.chunk_size:
                        # Split long sentence into smaller pieces
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk)
                                temp_chunk = word
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix for sentence embeddings.

        Args:
            embeddings: Sentence embeddings

        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

        return similarity_matrix

    def _find_semantic_boundaries(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Find semantic boundaries based on similarity drops.

        Args:
            similarity_matrix: Similarity matrix between sentences

        Returns:
            List of boundary indices
        """
        boundaries = [0]  # Start with first sentence

        for i in range(1, len(similarity_matrix) - 1):
            # Calculate average similarity with previous and next sentences
            prev_sim = similarity_matrix[i][i-1]
            next_sim = similarity_matrix[i][i+1]

            # If similarity drops below threshold, mark as boundary
            if prev_sim < self.similarity_threshold and next_sim < self.similarity_threshold:
                boundaries.append(i)

        boundaries.append(len(similarity_matrix))  # End with last sentence

        return boundaries

    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """
        Create chunks from semantic boundaries.

        Args:
            sentences: List of sentences
            boundaries: List of boundary indices

        Returns:
            List of chunks
        """
        chunks = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # Extract sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Check if chunk is too large
            if len(chunk_text) > self.chunk_size:
                # Split large chunk further
                sub_chunks = self._split_large_chunk(chunk_sentences)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return chunks

    def _split_large_chunk(self, sentences: List[str]) -> List[str]:
        """
        Split a large chunk into smaller pieces.

        Args:
            sentences: List of sentences in the chunk

        Returns:
            List of smaller chunks
        """
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

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

            # Get overlap from previous chunk (last few sentences)
            if len(previous_chunk) > self.chunk_overlap:
                # Try to get complete sentences for overlap
                prev_sentences = self._split_into_sentences(previous_chunk)
                overlap_text = ""

                # Add sentences from the end until we reach overlap size
                for j in range(len(prev_sentences) - 1, -1, -1):
                    sentence = prev_sentences[j]
                    if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                        overlap_text = sentence + " " + overlap_text if overlap_text else sentence
                    else:
                        break

                if overlap_text:
                    current_chunk = overlap_text + " " + current_chunk

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

        sentences = self._split_into_sentences(chunk)

        metadata.update({
            "chunker_type": "semantic",
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model,
            "use_embeddings": self.use_embeddings,
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean([len(s) for s in sentences]) if sentences else 0,
            "min_sentences_per_chunk": self.min_sentences_per_chunk
        })

        return metadata