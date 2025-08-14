from typing import List, Dict, Any
import re
from .interfaces import ChunkingInterface, Chunk
import logging

logger = logging.getLogger(__name__)


class FixedSizeChunking(ChunkingInterface):
    """Fixed-size text chunking strategy"""

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.separator = config.get('separator', '\n')

        logger.info(f"Initialized fixed-size chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into fixed-size chunks"""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not the last chunk, try to break at a natural boundary
            if end < len(text):
                # Look for the last separator within the chunk
                last_sep = text.rfind(self.separator, start, end)
                if last_sep > start:
                    end = last_sep + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = Chunk(
                    id=f"{metadata.get('id', 'doc')}_chunk_{chunk_id}",
                    text=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_start': start,
                        'chunk_end': end,
                        'chunk_size': len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_id += 1

            # Move start position, accounting for overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def get_chunking_info(self) -> Dict[str, Any]:
        return {
            'name': 'fixed-size-chunking',
            'strategy': 'fixed_size',
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'separator': self.separator
        }


class SemanticChunking(ChunkingInterface):
    """Semantic text chunking strategy based on natural boundaries"""

    def __init__(self, config: Dict[str, Any]):
        self.max_chunk_size = config.get('max_chunk_size', 1000)
        self.min_chunk_size = config.get('min_chunk_size', 200)
        self.boundary_patterns = config.get('boundary_patterns', [
            r'\n\n+',  # Multiple newlines
            r'\.\s+',  # Period followed by space
            r'!\s+',   # Exclamation followed by space
            r'\?\s+',  # Question mark followed by space
            r';\s+',   # Semicolon followed by space
            r':\s+',   # Colon followed by space
        ])

        logger.info(f"Initialized semantic chunking: max_size={self.max_chunk_size}, min_size={self.min_chunk_size}")

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into semantic chunks"""
        chunks = []
        chunk_id = 0
        current_chunk = ""
        current_start = 0

        # Split text into sentences/paragraphs
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            # If adding this sentence would exceed max size, create a chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = Chunk(
                        id=f"{metadata.get('id', 'doc')}_chunk_{chunk_id}",
                        text=current_chunk.strip(),
                        metadata={
                            **metadata,
                            'chunk_start': current_start,
                            'chunk_end': current_start + len(current_chunk),
                            'chunk_size': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                current_chunk = sentence
                current_start = text.find(sentence, current_start)
            else:
                if not current_chunk:
                    current_start = text.find(sentence)
                current_chunk += sentence

        # Add the last chunk if it meets minimum size
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                id=f"{metadata.get('id', 'doc')}_chunk_{chunk_id}",
                text=current_chunk.strip(),
                metadata={
                    **metadata,
                    'chunk_start': current_start,
                    'chunk_end': current_start + len(current_chunk),
                    'chunk_size': len(current_chunk)
                }
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using boundary patterns"""
        # Start with the text as one piece
        pieces = [text]

        # Apply each boundary pattern
        for pattern in self.boundary_patterns:
            new_pieces = []
            for piece in pieces:
                if len(piece) <= self.max_chunk_size:
                    new_pieces.append(piece)
                else:
                    # Split at boundaries
                    splits = re.split(pattern, piece)
                    new_pieces.extend(splits)
            pieces = new_pieces

        # Clean up and filter empty pieces
        return [piece.strip() for piece in pieces if piece.strip()]

    def get_chunking_info(self) -> Dict[str, Any]:
        return {
            'name': 'semantic-chunking',
            'strategy': 'semantic',
            'max_chunk_size': self.max_chunk_size,
            'min_chunk_size': self.min_chunk_size,
            'boundary_patterns': self.boundary_patterns
        }


class ChunkingFactory:
    """Factory for creating chunking instances based on configuration"""

    @staticmethod
    def create_chunking(config: Dict[str, Any]) -> ChunkingInterface:
        """Create chunking instance based on configuration"""
        chunking_type = config.get('type', 'fixed-size')

        if chunking_type == 'fixed-size':
            return FixedSizeChunking(config)
        elif chunking_type == 'semantic':
            return SemanticChunking(config)
        else:
            raise ValueError(f"Unknown chunking type: {chunking_type}")
