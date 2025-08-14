from typing import List, Dict, Any
import numpy as np
from .interfaces import RetrievalInterface, Chunk
import logging

logger = logging.getLogger(__name__)


class VectorSimilarityRetrieval(RetrievalInterface):
    """Vector similarity-based retrieval using cosine similarity"""

    def __init__(self, config: Dict[str, Any]):
        self.top_k = config.get('top_k', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.0)
        self.chunks = []
        self.embeddings = []

        logger.info(f"Initialized vector similarity retrieval: top_k={self.top_k}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks and their embeddings to the retrieval index"""
        for chunk in chunks:
            if chunk.embedding is not None:
                self.chunks.append(chunk)
                self.embeddings.append(chunk.embedding)
            else:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")

        logger.info(f"Added {len(self.chunks)} chunks to retrieval index")

    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve top-k most similar chunks"""
        if not self.chunks:
            logger.warning("No chunks in retrieval index")
            return []

        # Use instance top_k if not specified
        if top_k is None:
            top_k = self.top_k

        # For now, return random chunks (placeholder)
        # In a real implementation, you'd compute query embedding and find similarities
        logger.info(f"Retrieving top {top_k} chunks (placeholder implementation)")

        # Simple random selection for now
        import random
        selected_indices = random.sample(range(len(self.chunks)), min(top_k, len(self.chunks)))
        selected_chunks = [self.chunks[i] for i in selected_indices]

        return selected_chunks

    def get_retrieval_info(self) -> Dict[str, Any]:
        return {
            'name': 'vector-similarity-retrieval',
            'method': 'cosine_similarity',
            'chunks_indexed': len(self.chunks),
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }


class HybridRetrieval(RetrievalInterface):
    """Hybrid retrieval combining vector similarity and keyword matching"""

    def __init__(self, config: Dict[str, Any]):
        self.vector_weight = config.get('vector_weight', 0.7)
        self.keyword_weight = config.get('keyword_weight', 0.3)
        self.top_k = config.get('top_k', 5)
        self.chunks = []
        self.embeddings = []

        logger.info(f"Initialized hybrid retrieval: vector_weight={self.vector_weight}, keyword_weight={self.keyword_weight}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks and their embeddings to the retrieval index"""
        for chunk in chunks:
            if chunk.embedding is not None:
                self.chunks.append(chunk)
                self.embeddings.append(chunk.embedding)
            else:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")

        logger.info(f"Added {len(self.chunks)} chunks to retrieval index")

    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve chunks using hybrid approach"""
        if not self.chunks:
            logger.warning("No chunks in retrieval index")
            return []

        if top_k is None:
            top_k = self.top_k

        # Placeholder implementation - return random chunks
        logger.info(f"Retrieving top {top_k} chunks using hybrid approach (placeholder)")

        import random
        selected_indices = random.sample(range(len(self.chunks)), min(top_k, len(self.chunks)))
        selected_chunks = [self.chunks[i] for i in selected_indices]

        return selected_chunks

    def get_retrieval_info(self) -> Dict[str, Any]:
        return {
            'name': 'hybrid-retrieval',
            'method': 'hybrid',
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'chunks_indexed': len(self.chunks),
            'top_k': self.top_k
        }


class RetrievalFactory:
    """Factory for creating retrieval instances based on configuration"""

    @staticmethod
    def create_retrieval(config: Dict[str, Any]) -> RetrievalInterface:
        """Create retrieval instance based on configuration"""
        retrieval_type = config.get('type', 'vector-similarity')

        if retrieval_type == 'vector-similarity':
            return VectorSimilarityRetrieval(config)
        elif retrieval_type == 'hybrid':
            return HybridRetrieval(config)
        else:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")
