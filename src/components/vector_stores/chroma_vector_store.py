import os
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_vector_store import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store implementation for persistent vector storage.

    This class uses ChromaDB for storing and retrieving document embeddings
    with support for metadata filtering and persistent storage.
    """

    def __init__(self,
                 collection_name: str = "rag_documents",
                 persist_directory: Optional[str] = None,
                 embedding_function: Optional[Any] = None,
                 distance_metric: str = "cosine",
                 host: Optional[str] = None,
                 port: Optional[int] = None):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_function: Custom embedding function for ChromaDB
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
            host: Host for ChromaDB server (if using client mode)
            port: Port for ChromaDB server (if using client mode)
        """
        super().__init__()

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.distance_metric = distance_metric
        self.host = host
        self.port = port

        # Try to import chromadb
        try:
            import chromadb
            from chromadb.config import Settings
            self._chromadb = chromadb
            self._settings = Settings
            self.available = True
        except ImportError:
            self._chromadb = None
            self._settings = None
            self.available = False
            print("Warning: chromadb library not available. Please install it with: pip install chromadb")

        # Initialize client and collection
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        if not self.available:
            return

        try:
            # Create client
            if self.host and self.port:
                # Client mode
                self.client = self._chromadb.HttpClient(
                    host=self.host,
                    port=self.port
                )
            else:
                # Embedded mode
                if self.persist_directory:
                    # Persistent storage
                    self.client = self._chromadb.PersistentClient(
                        path=self.persist_directory
                    )
                else:
                    # In-memory storage
                    self.client = self._chromadb.EphemeralClient()

            # Create or get collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                print(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                metadata = {"hnsw:space": self.distance_metric}

                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata=metadata
                )
                print(f"Created new ChromaDB collection: {self.collection_name}")

        except Exception as e:
            print(f"Error initializing ChromaDB client: {e}")
            self.available = False
            self.client = None
            self.collection = None

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
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

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
            # Convert embeddings to list format
            embeddings_list = embeddings.tolist()

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Added {len(texts)} documents to ChromaDB collection")
            return ids

        except Exception as e:
            print(f"Error adding texts to ChromaDB: {e}")
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
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        if query_embedding.size == 0:
            return []

        try:
            # Convert embedding to list
            query_embedding_list = query_embedding.tolist()

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=filter_dict
            )

            # Process results
            search_results = []
            documents = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]

            for doc, distance, metadata in zip(documents, distances, metadatas):
                # Convert distance to similarity score
                if self.distance_metric == "cosine":
                    similarity = 1.0 - distance
                elif self.distance_metric == "l2":
                    similarity = 1.0 / (1.0 + distance)
                else:  # ip (inner product)
                    similarity = distance

                search_results.append((doc, similarity, metadata or {}))

            return search_results

        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            raise

    def get_all_documents(self) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Get all documents from the vector store.

        Returns:
            List of (text, embedding, metadata) tuples
        """
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        try:
            # Get all documents
            results = self.collection.get(
                include=["documents", "embeddings", "metadatas"]
            )

            documents = results.get('documents', [])
            embeddings = results.get('embeddings', [])
            metadatas = results.get('metadatas', [])

            # Convert to expected format
            all_docs = []
            for doc, emb, meta in zip(documents, embeddings, metadatas):
                embedding_array = np.array(emb) if emb else np.array([])
                all_docs.append((doc, embedding_array, meta or {}))

            return all_docs

        except Exception as e:
            print(f"Error getting all documents from ChromaDB: {e}")
            raise

    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful
        """
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        try:
            self.collection.delete(ids=ids)
            print(f"Deleted {len(ids)} documents from ChromaDB")
            return True

        except Exception as e:
            print(f"Error deleting documents from ChromaDB: {e}")
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
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        try:
            update_data = {"ids": ids}

            if texts:
                update_data["documents"] = texts

            if embeddings is not None:
                update_data["embeddings"] = embeddings.tolist()

            if metadatas:
                update_data["metadatas"] = metadatas

            self.collection.update(**update_data)
            print(f"Updated {len(ids)} documents in ChromaDB")
            return True

        except Exception as e:
            print(f"Error updating documents in ChromaDB: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self.available or self.collection is None:
            return {"error": "ChromaDB not available"}

        try:
            count = self.collection.count()

            stats = {
                "collection_name": self.collection_name,
                "document_count": count,
                "distance_metric": self.distance_metric,
                "persist_directory": self.persist_directory,
                "available": self.available
            }

            # Try to get additional metadata
            try:
                collection_metadata = self.collection.metadata
                if collection_metadata:
                    stats["collection_metadata"] = collection_metadata
            except Exception:
                pass

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
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        try:
            # Get all document IDs
            results = self.collection.get(include=[])
            ids = results.get('ids', [])

            if ids:
                self.collection.delete(ids=ids)
                print(f"Cleared {len(ids)} documents from ChromaDB collection")
            else:
                print("Collection is already empty")

            return True

        except Exception as e:
            print(f"Error clearing ChromaDB collection: {e}")
            return False

    def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create or rebuild index (ChromaDB handles indexing automatically).

        Args:
            index_params: Optional index parameters (not used in ChromaDB)

        Returns:
            True (ChromaDB handles indexing automatically)
        """
        print("ChromaDB handles indexing automatically")
        return True

    def backup_collection(self, backup_path: str) -> bool:
        """
        Backup the collection to a file.

        Args:
            backup_path: Path to save the backup

        Returns:
            True if successful
        """
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        try:
            # Get all data
            results = self.collection.get(
                include=["documents", "embeddings", "metadatas"]
            )

            # Save to file
            import json
            backup_data = {
                "collection_name": self.collection_name,
                "distance_metric": self.distance_metric,
                "data": results
            }

            with open(backup_path, 'w') as f:
                json.dump(backup_data, f)

            print(f"Collection backed up to: {backup_path}")
            return True

        except Exception as e:
            print(f"Error backing up collection: {e}")
            return False

    def restore_collection(self, backup_path: str) -> bool:
        """
        Restore collection from a backup file.

        Args:
            backup_path: Path to the backup file

        Returns:
            True if successful
        """
        if not self.available or self.collection is None:
            raise RuntimeError("ChromaDB not available")

        try:
            import json

            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            data = backup_data.get("data", {})
            documents = data.get("documents", [])
            embeddings = data.get("embeddings", [])
            metadatas = data.get("metadatas", [])
            ids = data.get("ids", [])

            if documents:
                # Clear existing collection
                self.clear_collection()

                # Add restored data
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )

                print(f"Restored {len(documents)} documents from backup")

            return True

        except Exception as e:
            print(f"Error restoring collection: {e}")
            return False