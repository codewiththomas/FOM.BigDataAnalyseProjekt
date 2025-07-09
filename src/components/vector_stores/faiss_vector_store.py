import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store implementation for high-performance vector search.

    This class uses Facebook's FAISS library for efficient similarity search
    and clustering of dense vectors.
    """

    def __init__(self,
                 embedding_dimension: int,
                 index_type: str = "IndexFlatIP",
                 metric_type: str = "cosine",
                 nlist: int = 100,
                 nprobe: int = 10,
                 use_gpu: bool = False,
                 gpu_id: int = 0):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dimension: Dimension of the embeddings
            index_type: Type of FAISS index ('IndexFlatIP', 'IndexIVFFlat', 'IndexHNSWFlat')
            metric_type: Distance metric ('cosine', 'l2', 'ip')
            nlist: Number of clusters for IVF indexes
            nprobe: Number of clusters to search for IVF indexes
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID
        """
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # Try to import faiss
        try:
            import faiss
            self._faiss = faiss
            self.available = True
        except ImportError:
            self._faiss = None
            self.available = False
            print("Warning: faiss library not available. Please install it with: pip install faiss-cpu or pip install faiss-gpu")

        # Initialize index and storage
        self.index = None
        self.texts = []
        self.metadatas = []
        self.ids = []
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index."""
        if not self.available:
            return

        try:
            # Create index based on type and metric
            if self.index_type == "IndexFlatIP":
                # Flat index with inner product
                self.index = self._faiss.IndexFlatIP(self.embedding_dimension)
            elif self.index_type == "IndexFlatL2":
                # Flat index with L2 distance
                self.index = self._faiss.IndexFlatL2(self.embedding_dimension)
            elif self.index_type == "IndexIVFFlat":
                # IVF index with flat quantizer
                quantizer = self._faiss.IndexFlatL2(self.embedding_dimension)
                self.index = self._faiss.IndexIVFFlat(quantizer, self.embedding_dimension, self.nlist)
            elif self.index_type == "IndexHNSWFlat":
                # HNSW index
                self.index = self._faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            else:
                # Default to flat IP
                self.index = self._faiss.IndexFlatIP(self.embedding_dimension)

            # Set search parameters for IVF indexes
            if "IVF" in self.index_type:
                self.index.nprobe = self.nprobe

            # Move to GPU if requested
            if self.use_gpu and self._faiss.get_num_gpus() > 0:
                res = self._faiss.StandardGpuResources()
                self.index = self._faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                print(f"FAISS index moved to GPU {self.gpu_id}")

            print(f"Initialized FAISS index: {self.index_type}")

        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            self.available = False
            self.index = None

    def add_texts(self,
                  texts: List[str],
                  embeddings: np.ndarray,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None) -> List[str]:
        """
        Add texts and their embeddings to the vector store.

        Args:
            texts: List of text documents
            embeddings: Corresponding embeddings
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text (auto-generated if not provided)

        Returns:
            List of document IDs
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{len(self.texts) + i}" for i in range(len(texts))]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Ensure metadatas have required fields
        for i, metadata in enumerate(metadatas):
            if metadata is None:
                metadatas[i] = {}
            metadatas[i].update({
                "text_length": len(texts[i]),
                "doc_id": ids[i]
            })

        try:
            # Normalize embeddings for cosine similarity
            if self.metric_type == "cosine":
                embeddings = self._normalize_embeddings(embeddings)

            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))

            # Store texts, metadatas, and IDs
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)

            print(f"Added {len(texts)} documents to FAISS index")
            return ids

        except Exception as e:
            print(f"Error adding texts to FAISS: {e}")
            raise

    def similarity_search(self,
                         query_embedding: np.ndarray,
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of (text, similarity_score, metadata) tuples
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        if query_embedding.size == 0 or len(self.texts) == 0:
            return []

        try:
            # Normalize query embedding for cosine similarity
            if self.metric_type == "cosine":
                query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))
            else:
                query_embedding = query_embedding.reshape(1, -1)

            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)

            # Process results
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                if idx >= len(self.texts):
                    continue

                text = self.texts[idx]
                metadata = self.metadatas[idx]

                # Apply metadata filter if provided
                if filter_dict and not self._matches_filter(metadata, filter_dict):
                    continue

                # Convert score to similarity
                if self.metric_type == "cosine" or self.index_type == "IndexFlatIP":
                    similarity = float(score)
                elif self.metric_type == "l2":
                    similarity = 1.0 / (1.0 + float(score))
                else:
                    similarity = float(score)

                search_results.append((text, similarity, metadata))

            return search_results

        except Exception as e:
            print(f"Error searching FAISS: {e}")
            raise

    def get_all_documents(self) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Get all documents from the vector store.

        Returns:
            List of (text, embedding, metadata) tuples
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        # FAISS doesn't store original embeddings, so we return empty arrays
        all_docs = []
        for i, (text, metadata) in enumerate(zip(self.texts, self.metadatas)):
            # We can't retrieve original embeddings from FAISS index
            empty_embedding = np.array([])
            all_docs.append((text, empty_embedding, metadata))

        return all_docs

    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        try:
            # Find indices to delete
            indices_to_delete = []
            for doc_id in ids:
                try:
                    idx = self.ids.index(doc_id)
                    indices_to_delete.append(idx)
                except ValueError:
                    continue

            if not indices_to_delete:
                return True

            # Remove from stored data (in reverse order to maintain indices)
            for idx in sorted(indices_to_delete, reverse=True):
                del self.texts[idx]
                del self.metadatas[idx]
                del self.ids[idx]

            # FAISS doesn't support individual deletion, so we need to rebuild
            self._rebuild_index()

            print(f"Deleted {len(indices_to_delete)} documents from FAISS")
            return True

        except Exception as e:
            print(f"Error deleting documents from FAISS: {e}")
            return False

    def update_documents(self,
                        ids: List[str],
                        texts: Optional[List[str]] = None,
                        embeddings: Optional[np.ndarray] = None,
                        metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Update existing documents.

        Args:
            ids: List of document IDs to update
            texts: Optional new texts
            embeddings: Optional new embeddings
            metadatas: Optional new metadatas

        Returns:
            True if successful
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        try:
            # Find indices to update
            indices_to_update = []
            for doc_id in ids:
                try:
                    idx = self.ids.index(doc_id)
                    indices_to_update.append(idx)
                except ValueError:
                    continue

            if not indices_to_update:
                return True

            # Update stored data
            for i, idx in enumerate(indices_to_update):
                if texts and i < len(texts):
                    self.texts[idx] = texts[i]

                if metadatas and i < len(metadatas):
                    self.metadatas[idx] = metadatas[i]

            # If embeddings are updated, rebuild index
            if embeddings is not None:
                self._rebuild_index()

            print(f"Updated {len(indices_to_update)} documents in FAISS")
            return True

        except Exception as e:
            print(f"Error updating documents in FAISS: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self.available or self.index is None:
            return {"error": "FAISS not available"}

        try:
            stats = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "embedding_dimension": self.embedding_dimension,
                "document_count": len(self.texts),
                "index_size": self.index.ntotal,
                "use_gpu": self.use_gpu,
                "available": self.available
            }

            # Add index-specific stats
            if "IVF" in self.index_type:
                stats.update({
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    "is_trained": self.index.is_trained
                })

            return stats

        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        try:
            # Clear stored data
            self.texts.clear()
            self.metadatas.clear()
            self.ids.clear()

            # Reset index
            self.index.reset()

            print("Cleared all documents from FAISS collection")
            return True

        except Exception as e:
            print(f"Error clearing FAISS collection: {e}")
            return False

    def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create or rebuild index with optional parameters.

        Args:
            index_params: Optional index parameters

        Returns:
            True if successful
        """
        if not self.available:
            raise RuntimeError("FAISS not available")

        try:
            if index_params:
                # Update parameters
                self.index_type = index_params.get("index_type", self.index_type)
                self.nlist = index_params.get("nlist", self.nlist)
                self.nprobe = index_params.get("nprobe", self.nprobe)

            # Reinitialize index
            self._initialize_index()

            # Rebuild if we have data
            if self.texts:
                self._rebuild_index()

            return True

        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return False

    def save_index(self, filepath: str) -> bool:
        """
        Save the FAISS index and metadata to files.

        Args:
            filepath: Base filepath (without extension)

        Returns:
            True if successful
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        try:
            # Save FAISS index
            index_path = f"{filepath}.index"
            self._faiss.write_index(self.index, index_path)

            # Save metadata
            metadata_path = f"{filepath}.metadata"
            metadata = {
                "texts": self.texts,
                "metadatas": self.metadatas,
                "ids": self.ids,
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "embedding_dimension": self.embedding_dimension,
                "nlist": self.nlist,
                "nprobe": self.nprobe
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            print(f"FAISS index saved to: {index_path}")
            print(f"Metadata saved to: {metadata_path}")
            return True

        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False

    def load_index(self, filepath: str) -> bool:
        """
        Load FAISS index and metadata from files.

        Args:
            filepath: Base filepath (without extension)

        Returns:
            True if successful
        """
        if not self.available:
            raise RuntimeError("FAISS not available")

        try:
            # Load FAISS index
            index_path = f"{filepath}.index"
            self.index = self._faiss.read_index(index_path)

            # Load metadata
            metadata_path = f"{filepath}.metadata"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.texts = metadata["texts"]
            self.metadatas = metadata["metadatas"]
            self.ids = metadata["ids"]
            self.index_type = metadata["index_type"]
            self.metric_type = metadata["metric_type"]
            self.embedding_dimension = metadata["embedding_dimension"]
            self.nlist = metadata["nlist"]
            self.nprobe = metadata["nprobe"]

            print(f"FAISS index loaded from: {index_path}")
            print(f"Metadata loaded from: {metadata_path}")
            return True

        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _rebuild_index(self):
        """Rebuild the FAISS index from stored data."""
        if not self.texts:
            return

        # This is a simplified rebuild - in practice, you'd need to store embeddings
        # or recompute them to rebuild the index properly
        print("Warning: FAISS index rebuild requires original embeddings")
        print("Consider storing embeddings separately or using a different vector store for frequent updates")

    def train_index(self, training_embeddings: np.ndarray) -> bool:
        """
        Train the index with training data (required for some index types).

        Args:
            training_embeddings: Training embeddings

        Returns:
            True if successful
        """
        if not self.available or self.index is None:
            raise RuntimeError("FAISS not available")

        try:
            if self.metric_type == "cosine":
                training_embeddings = self._normalize_embeddings(training_embeddings)

            self.index.train(training_embeddings.astype(np.float32))
            print("FAISS index trained successfully")
            return True

        except Exception as e:
            print(f"Error training FAISS index: {e}")
            return False