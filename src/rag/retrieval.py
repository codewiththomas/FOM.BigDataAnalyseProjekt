from typing import List, Dict, Any
import numpy as np
from interfaces import RetrievalInterface, Chunk
import logging

logger = logging.getLogger(__name__)


class VectorSimilarityRetrieval(RetrievalInterface):
    """
    Vektorähnlichkeitsbasierte Retrieval-Methode, welche die Ähnlichkeit zwischen Vektoren und einer Abfrage mit
    Cosinus-Ähnlichkeit berechnet
    """

    def __init__(self, config: Dict[str, Any]):
        self.top_k = config.get('top_k', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.0)
        self.chunks = []
        self.embeddings = []
        self._embedding_model = None # muss vom Typ EmbeddingInterface sein

        logger.info(f"Vector similarity retrieval: top_k={self.top_k}, threshold={self.similarity_threshold}")

    def set_embedding_model(self, embedding_model):
        """Link the same embedding model used for chunks to the retriever."""
        self._embedding_model = embedding_model
        if embedding_model is None:
            logger.warning("Embedding model set to None - retrieval may fail")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks and their embeddings to the retrieval index"""
        self.chunks = chunks
        self.embeddings = embeddings

        # Convert embeddings to numpy array for efficient computation
        if embeddings:
            emb_dims = set(len(emb) for emb in embeddings)
            if len(emb_dims) > 1:
                raise ValueError(f"Inconsistent embedding dimensions: {emb_dims}")

            self.embeddings_array = np.array(embeddings)
            logger.info(f"Added {len(chunks)} chunks to retrieval index with {self.embeddings_array.shape[1]}-dimensional embeddings")
        else:
            self.embeddings_array = np.array([])
            logger.warning("No embeddings provided")

    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve top-k most similar chunks using cosine similarity"""
        if not self.chunks or len(self.embeddings_array) == 0:
            logger.warning("No chunks or embeddings available for retrieval")
            return []

        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.top_k

        try:
            # Generate query embedding (this should be done by the embedding model)
            # For now, we'll use a simple approach - in production, you'd use the same embedding model
            query_embedding = self._get_query_embedding(query)

            if query_embedding is None:
                logger.warning("Could not generate query embedding, returning random chunks")
                return self._get_random_chunks(top_k)

            # Calculate cosine similarities
            similarities = self._calculate_cosine_similarities(query_embedding)

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Filter by similarity threshold
            top_indices = [idx for idx in top_indices if similarities[idx] >= self.similarity_threshold]

            # Return chunks with similarity scores
            retrieved_chunks = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                # Add similarity score to chunk metadata
                chunk.metadata['similarity_score'] = float(similarities[idx])
                retrieved_chunks.append(chunk)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks with similarities: {[f'{c.metadata.get("similarity_score", 0):.3f}' for c in retrieved_chunks]}")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error in vector similarity retrieval: {e}")
            logger.warning("Falling back to random chunk selection")
            return self._get_random_chunks(top_k)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query using the same model as for chunks."""
        if not hasattr(self, "_embedding_model") or self._embedding_model is None:
            logger.error("No embedding model linked to retrieval; cannot embed query.")
            raise RuntimeError("Embedding model must be set before retrieval")  # ← FIX
            # return np.ndarray([], dtype=float) # leeres Array wird zurückgegeben -> Problem bei Berechnung cosine similarity (Division durch 0) -> Fallback auf Random Chunks
        vec = self._embedding_model.embed([query])[0]  # list[float]
        return np.array(vec, dtype=float)

    def _calculate_cosine_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between query and all chunk embeddings"""
        # Normalize embeddings for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(self.embeddings_array))

        query_normalized = query_embedding / query_norm

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings_array, query_normalized)

        # Normalize chunk embeddings
        chunk_norms = np.linalg.norm(self.embeddings_array, axis=1)
        chunk_norms[chunk_norms == 0] = 1  # Avoid division by zero

        # Normalize and calculate final similarities
        similarities = similarities / chunk_norms

        return similarities

    def _get_random_chunks(self, top_k: int) -> List[Chunk]:
        """Fallback method to get random chunks"""
        import random
        selected_chunks = random.sample(self.chunks, min(top_k, len(self.chunks)))
        for chunk in selected_chunks:
            chunk.metadata['similarity_score'] = 0.0  # Random selection
        return selected_chunks

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': 'vector-similarity-retrieval',
            'method': 'cosine_similarity',
            'chunks_indexed': len(self.chunks),
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'embedding_dimensions': self.embeddings_array.shape[1] if self.embeddings_array.size else 0
        }

    def get_retrieval_info(self) -> Dict[str, Any]:
        """Get retrieval information (alias for get_model_info for compatibility)"""
        return self.get_model_info()


class HybridRetrieval(RetrievalInterface):
    """
    Hybrid retrieval combining vector and keyword search
    """

    def __init__(self, config: Dict[str, Any]):
        self.vector_weight = config.get('vector_weight', 0.7)
        self.keyword_weight = config.get('keyword_weight', 0.3)
        self.top_k = config.get('top_k', 5)
        self.chunks = []
        self.embeddings = []

        logger.info(f"Hybrid retrieval: vector_weight={self.vector_weight}, keyword_weight={self.keyword_weight}")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks and their embeddings to the retrieval index"""
        self.chunks = chunks
        self.embeddings = embeddings

        # Convert embeddings to numpy array
        if embeddings:
            self.embeddings_array = np.array(embeddings)
            logger.info(f"Added {len(chunks)} chunks to retrieval index")
        else:
            self.embeddings_array = np.array([])

    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve chunks using hybrid approach"""
        if not self.chunks or len(self.embeddings_array) == 0:
            return []

        if top_k is None:
            top_k = self.top_k

        try:
            # Vector similarity score
            vector_scores = self._get_vector_scores(query)

            # Keyword relevance score
            keyword_scores = self._get_keyword_scores(query)

            # Combine scores with weights
            combined_scores = (self.vector_weight * vector_scores +
                             self.keyword_weight * keyword_scores)

            # Get top-k indices
            top_indices = np.argsort(combined_scores)[::-1][:top_k]

            # Return chunks with combined scores
            retrieved_chunks = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                chunk.metadata['combined_score'] = float(combined_scores[idx])
                chunk.metadata['vector_score'] = float(vector_scores[idx])
                chunk.metadata['keyword_score'] = float(keyword_scores[idx])
                retrieved_chunks.append(chunk)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks using hybrid approach")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return self._get_random_chunks(top_k)

    def _get_vector_scores(self, query: str) -> np.ndarray:
        """Get vector similarity scores"""
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return np.zeros(len(self.chunks))

        similarities = self._calculate_cosine_similarities(query_embedding)
        return similarities

    def _get_keyword_scores(self, query: str) -> np.ndarray:
        """Get keyword relevance scores"""
        query_words = set(query.lower().split())
        scores = []

        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            if not query_words:
                scores.append(0.0)
                continue

            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))

            if union == 0:
                scores.append(0.0)
            else:
                scores.append(intersection / union)

        return np.array(scores)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query"""
        if not self.embeddings_array.size:
            return None

        embedding_dim = self.embeddings_array.shape[1]
        import hashlib
        hash_obj = hashlib.md5(query.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(embedding_dim):
            if i < len(hash_bytes):
                embedding.append(float(hash_bytes[i]) / 255.0)
            else:
                embedding.append(0.0)

        return np.array(embedding)

    def _calculate_cosine_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities"""
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(self.embeddings_array))

        query_normalized = query_embedding / query_norm
        similarities = np.dot(self.embeddings_array, query_normalized)

        chunk_norms = np.linalg.norm(self.embeddings_array, axis=1)
        chunk_norms[chunk_norms == 0] = 1

        similarities = similarities / chunk_norms
        return similarities

    def _get_random_chunks(self, top_k: int) -> List[Chunk]:
        """Fallback method"""
        import random
        selected_chunks = random.sample(self.chunks, min(top_k, len(self.chunks)))
        for chunk in selected_chunks:
            chunk.metadata['combined_score'] = 0.0
        return selected_chunks

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': 'hybrid-retrieval',
            'method': 'hybrid',
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'chunks_indexed': len(self.chunks),
            'top_k': self.top_k
        }

    def get_retrieval_info(self) -> Dict[str, Any]:
        """Get retrieval information"""
        return self.get_model_info()


# class FAISSRetrieval(RetrievalInterface):
#     """
#     Implementierung von FAISS als Retrieval-Methode
#     """
#     raise NotImplementedError("FAISSRetrieval ist noch nicht implementiert")


class RetrievalFactory:
    """
    Factory-Klasse, welche eine Instanz der gewünschten Retrieval-Methode basierend auf der Konfiguration erstellt
    """

    @staticmethod
    def create_retrieval(config: Dict[str, Any]) -> RetrievalInterface:
        """
        Create retrieval instance based on configuration
        """
        retrieval_type = config.get('type', 'vector-similarity')

        if retrieval_type == 'vector-similarity':
            return VectorSimilarityRetrieval(config)
        elif retrieval_type == 'hybrid':
            return HybridRetrieval(config)
        # elif retrieval_type == 'faiss':
        #     return FAISSRetrieval(config)
        else:
            raise ValueError(f"Unbekannte Retrieval-Methode: {retrieval_type}")
