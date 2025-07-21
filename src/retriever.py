"""
Retriever module for FAISS-based vector search and retrieval.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm  # type: ignore


class FAISSRetriever:
    """FAISS-based retriever for semantic search."""

    def __init__(self, embedding_model_name: str = "text-embedding-3-small", dimension: int = 1536):
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.embedding_model: Optional[SentenceTransformer] = None

        # Initialize embedding model based on type
        if "text-embedding" in embedding_model_name:
            # OpenAI embedding model - will be handled differently
            self.use_openai_embeddings = True
        else:
            # Use sentence-transformers
            self.use_openai_embeddings = False
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()

    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI()

            # Process in batches to handle API limits
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings, dtype=np.float32)

        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")
            print("Falling back to sentence-transformers...")
            # Fallback to sentence-transformers
            if not self.embedding_model:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        if self.use_openai_embeddings:
            return self._get_openai_embeddings(texts)
        else:
            return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from document chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        self.chunks = chunks

        print(f"Building FAISS index for {len(chunks)} chunks...")

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Get embeddings
        print("Computing embeddings...")
        embeddings = self._get_embeddings(texts)

        # Update dimension if needed
        if embeddings.shape[1] != self.dimension:
            self.dimension = embeddings.shape[1]

        # Create FAISS index
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        self.index.add(embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query.

        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of relevant chunk dictionaries with similarity scores
        """
        if not self.index or not self.chunks:
            raise ValueError("Index not built. Call build_index() first.")

        # Get query embedding
        query_embedding = self._get_embeddings([query])
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold:
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(score)
                results.append(chunk)

        return results

    def save_index(self, index_path: str) -> None:
        """
        Save FAISS index and chunks to disk.

        Args:
            index_path: Directory path to save index
        """
        if not self.index or not self.chunks:
            raise ValueError("No index to save. Build index first.")

        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_dir / "faiss_index.bin"))

        # Save chunks and metadata
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # Save metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "num_chunks": len(self.chunks),
            "use_openai_embeddings": self.use_openai_embeddings
        }

        with open(index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"Index saved to {index_path}")

    def load_index(self, index_path: str) -> bool:
        """
        Load FAISS index and chunks from disk.

        Args:
            index_path: Directory path containing saved index

        Returns:
            True if loaded successfully, False otherwise
        """
        index_dir = Path(index_path)

        if not index_dir.exists():
            print(f"Index path {index_path} does not exist")
            return False

        try:
            # Load FAISS index
            index_file = index_dir / "faiss_index.bin"
            if not index_file.exists():
                print(f"FAISS index file not found at {index_file}")
                return False

            self.index = faiss.read_index(str(index_file))

            # Load chunks
            chunks_file = index_dir / "chunks.pkl"
            if chunks_file.exists():
                with open(chunks_file, "rb") as f:
                    self.chunks = pickle.load(f)

            # Load metadata
            metadata_file = index_dir / "metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)
                    self.embedding_model_name = metadata.get("embedding_model", self.embedding_model_name)
                    self.dimension = metadata.get("dimension", self.dimension)
                    self.use_openai_embeddings = metadata.get("use_openai_embeddings", True)

            print(f"Index loaded successfully: {len(self.chunks)} chunks")
            return True

        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "num_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "use_openai_embeddings": self.use_openai_embeddings
        }


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader, DocumentChunker

    # Create sample chunks for testing
    sample_chunks = [
        {"text": "Die DSGVO ist eine EU-Verordnung zum Datenschutz.", "source": "test.txt"},
        {"text": "Personenbezogene Daten müssen geschützt werden.", "source": "test.txt"},
        {"text": "Unternehmen müssen Datenschutz-Folgenabschätzungen durchführen.", "source": "test.txt"}
    ]

    # Test retriever
    retriever = FAISSRetriever(embedding_model_name="all-MiniLM-L6-v2")
    retriever.build_index(sample_chunks)

    # Test search
    results = retriever.search("Was ist die DSGVO?", top_k=2)
    print(f"Found {len(results)} results")
    for result in results:
        print(f"Score: {result['similarity_score']:.3f} - {result['text'][:100]}...")

    print("Retriever test complete!")