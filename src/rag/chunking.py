from typing import List, Dict, Any
import re
from interfaces import ChunkingInterface, Chunk
import logging

logger = logging.getLogger(__name__)


class FixedSizeChunking(ChunkingInterface):
    """Fixed-size chunking strategy"""

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.separator = config.get('separator', '\n')

        logger.info(f"Fixed-size chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk a list of documents"""
        all_chunks = []
        chunk_id = 0

        for doc in documents:
            doc_chunks = self.chunk(doc['text'], doc.get('metadata', {}), chunk_id)
            all_chunks.extend(doc_chunks)
            chunk_id += len(doc_chunks)

        return all_chunks

    def chunk(self, text: str, metadata: Dict[str, Any] = None, start_id: int = 0) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap"""
        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = start_id

        while start < len(text):
            end = start + self.chunk_size

            # Extract chunk text
            chunk_text = text[start:end]

            # Create chunk with metadata
            chunk = Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata={
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_size': len(chunk_text),
                    'overlap': self.chunk_overlap if start > 0 else 0,
                    **(metadata or {})
                }
            )

            chunks.append(chunk)
            chunk_id += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

            # Avoid infinite loop for very short texts
            if start >= len(text):
                break

        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
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
    """Semantic chunking strategy that tries to split at natural boundaries"""

    def __init__(self, config: Dict[str, Any]):
        self.min_chunk_size = config.get('min_chunk_size', 500)
        self.max_chunk_size = config.get('max_chunk_size', 1500)
        self.separator = config.get('separator', '\n\n')

        logger.info(f"Semantic chunking: min={self.min_chunk_size}, max={self.max_chunk_size}")

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk a list of documents"""
        all_chunks = []
        chunk_id = 0

        for doc in documents:
            doc_chunks = self.chunk(doc['text'], doc.get('metadata', {}), chunk_id)
            all_chunks.extend(doc_chunks)
            chunk_id += len(doc_chunks)

        return all_chunks

    def chunk(self, text: str, metadata: Dict[str, Any] = None, start_id: int = 0) -> List[Chunk]:
        """Split text at natural boundaries while respecting size constraints"""
        if not text:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = text.split(self.separator)
        chunks = []
        current_chunk = ""
        current_metadata = metadata or {}
        chunk_id = start_id

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = Chunk(
                        id=f"chunk_{chunk_id}",
                        text=current_chunk.strip(),
                        metadata={
                            'paragraphs': i,
                            'chunk_size': len(current_chunk),
                            **(current_metadata or {})
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                # Start new chunk
                current_chunk = paragraph
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk if it meets minimum size
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                id=f"chunk_{chunk_id}",
                text=current_chunk.strip(),
                metadata={
                    'paragraphs': len(paragraphs),
                    'chunk_size': len(current_chunk),
                    **(current_metadata or {})
                }
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
        return chunks

    def get_chunking_info(self) -> Dict[str, Any]:
        return {
            'name': 'semantic-chunking',
            'strategy': 'semantic',
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'separator': self.separator
        }


class RecursiveChunking(ChunkingInterface):
    """Recursive chunking strategy that tries multiple separators in hierarchical order"""

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 100)
        self.separators = config.get('separators', ['\n\n', '\n', '. ', ' '])

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk a list of documents"""
        all_chunks = []
        chunk_id = 0

        for doc in documents:
            doc_chunks = self.chunk(doc['text'], doc.get('metadata', {}), chunk_id)
            all_chunks.extend(doc_chunks)
            chunk_id += len(doc_chunks)

        return all_chunks

    def chunk(self, text: str, metadata: Dict[str, Any] = None, start_id: int = 0) -> List[Chunk]:
        """Recursively split text using hierarchical separators"""
        if not text:
            return []

        splits = self._split_text(text)
        chunks = []

        for i, split in enumerate(splits):
            if split.strip():
                chunk = Chunk(
                    id=f"chunk_{start_id + i}",
                    text=split.strip(),
                    metadata={'chunk_size': len(split), **(metadata or {})}
                )
                chunks.append(chunk)

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text recursively using separators"""
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator
        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""

                for part in parts:
                    test = current + sep + part if current else part

                    if len(test) <= self.chunk_size:
                        current = test
                    else:
                        if current:
                            chunks.append(current)
                            # Add overlap
                            if self.chunk_overlap and len(current) > self.chunk_overlap:
                                current = current[-self.chunk_overlap:] + sep + part
                            else:
                                current = part
                        else:
                            # Part too big, split recursively
                            chunks.extend(self._split_text(part))
                            current = ""

                if current:
                    chunks.append(current)
                return chunks

        # Force split if no separators work
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def get_chunking_info(self) -> Dict[str, Any]:
        return {
            'name': 'recursive-chunking',
            'strategy': 'recursive',
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'separators': self.separators
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
        elif chunking_type == 'recursive':
            return RecursiveChunking(config)
        else:
            raise ValueError(f"Unknown chunking type: {chunking_type}")