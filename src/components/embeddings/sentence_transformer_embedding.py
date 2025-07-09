import os
import numpy as np
from typing import List, Optional, Dict, Any
from .base_embedding import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Sentence Transformer embedding implementation using local models.

    This class uses the sentence-transformers library to generate embeddings
    using pre-trained models that can run locally without API calls.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True,
                 batch_size: int = 32,
                 show_progress_bar: bool = False,
                 cache_folder: Optional[str] = None):
        """
        Initialize the Sentence Transformer embedding.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize embeddings to unit length
            batch_size: Batch size for processing multiple texts
            show_progress_bar: Whether to show progress bar during encoding
            cache_folder: Custom cache folder for models
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.cache_folder = cache_folder

        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_transformer = SentenceTransformer
            self.available = True
        except ImportError:
            self._sentence_transformer = None
            self.available = False
            print("Warning: sentence-transformers library not available. Please install it with: pip install sentence-transformers")

        # Initialize the model
        self.model = None
        self._embedding_dimension = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not self.available:
            return

        try:
            # Set cache folder if specified
            if self.cache_folder:
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = self.cache_folder

            # Initialize model
            self.model = self._sentence_transformer(
                self.model_name,
                device=self.device
            )

            # Get embedding dimension
            self._embedding_dimension = self.model.get_sentence_embedding_dimension()

            print(f"Loaded SentenceTransformer model: {self.model_name}")
            print(f"Embedding dimension: {self._embedding_dimension}")
            print(f"Device: {self.model.device}")

        except Exception as e:
            print(f"Error initializing SentenceTransformer model: {e}")
            self.available = False
            self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.available or self.model is None:
            raise RuntimeError("SentenceTransformer model not available")

        if not texts:
            return np.array([])

        # Clean texts
        cleaned_texts = [text.strip() for text in texts if text.strip()]

        if not cleaned_texts:
            return np.array([])

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )

            return embeddings

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query string to embed

        Returns:
            Numpy array embedding
        """
        if not query.strip():
            return np.array([])

        embeddings = self.embed_texts([query])
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Embedding dimension
        """
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()

        # Default dimensions for common models
        dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-distilroberta-v1": 768,
            "all-roberta-large-v1": 1024,
            "paraphrase-MiniLM-L6-v2": 384,
            "paraphrase-mpnet-base-v2": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768
        }

        return dimension_map.get(self.model_name, 384)

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0

        # Normalize embeddings if not already normalized
        if not self.normalize_embeddings:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "device": str(self.model.device) if self.model else "unknown",
            "available": self.available,
            "model_type": "sentence_transformer"
        }

        if self.model is not None:
            # Add model-specific information
            try:
                info.update({
                    "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
                    "model_card": getattr(self.model, 'model_card', {}),
                    "similarity_fn_name": getattr(self.model, 'similarity_fn_name', 'unknown')
                })
            except Exception:
                pass

        return info

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to a file.

        Args:
            embeddings: Embeddings to save
            filepath: Path to save the embeddings
        """
        try:
            np.save(filepath, embeddings)
            print(f"Embeddings saved to: {filepath}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            raise

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from a file.

        Args:
            filepath: Path to load embeddings from

        Returns:
            Loaded embeddings
        """
        try:
            embeddings = np.load(filepath)
            print(f"Embeddings loaded from: {filepath}")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """
        Get list of popular sentence transformer models.

        Returns:
            List of available model names
        """
        return [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
            "all-distilroberta-v1",
            "all-roberta-large-v1",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-mpnet-base-v2",
            "paraphrase-distilroberta-base-v1",
            "distiluse-base-multilingual-cased",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]

    def benchmark_model(self, test_texts: List[str]) -> Dict[str, Any]:
        """
        Benchmark the model performance on test texts.

        Args:
            test_texts: List of texts to benchmark on

        Returns:
            Dictionary with benchmark results
        """
        if not self.available or self.model is None:
            return {"error": "Model not available"}

        import time

        # Benchmark embedding generation
        start_time = time.time()
        embeddings = self.embed_texts(test_texts)
        end_time = time.time()

        total_time = end_time - start_time
        texts_per_second = len(test_texts) / total_time if total_time > 0 else 0

        return {
            "model_name": self.model_name,
            "num_texts": len(test_texts),
            "total_time": total_time,
            "texts_per_second": texts_per_second,
            "avg_time_per_text": total_time / len(test_texts) if test_texts else 0,
            "embedding_shape": embeddings.shape,
            "memory_usage": f"{embeddings.nbytes / 1024 / 1024:.2f} MB"
        }