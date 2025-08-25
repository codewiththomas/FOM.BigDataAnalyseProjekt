from typing import List, Dict, Any, Union
import openai
from interfaces import EmbeddingInterface
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingInterface):
    """OpenAI API-based embedding implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'text-embedding-ada-002')

        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            logger.warning("No OpenAI API key provided, using environment variable")
            self.client = openai.OpenAI()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 100

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            # Return zero vectors as fallback
            # return [[0.0] * 1536] * len(texts)  # OpenAI embeddings are 1536-dimensional
            dims = self.get_model_info().get('dimensions', 1536)  # ← Dynamisch!
            return [[0.0] * dims] * len(texts)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': f'openai-{self.model}',
            'provider': 'openai',
            'model': self.model,
            'dimensions': 1536
        }


class SentenceTransformersEmbedding(EmbeddingInterface):
    """Sentence-transformers based embedding implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'all-MiniLM-L6-v2')
        self.device = config.get('device', 'cpu')
        # self.batch_size = 32  # ← vorher evtl. implizit klein
        self.batch_size = config.get('batch_size', 128 if self.device == 'cuda' else 32)  # ← neu

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded sentence-transformers model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers"""
        try:
            # embeddings = self.model.encode(texts, convert_to_tensor=False)  # ← vorher
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                batch_size=self.batch_size,  # ← neu: nutzt größere Batches auf GPU
                show_progress_bar=True  # ← neu: besseres Feedback
            )
            # Ensure we return a list of lists of floats
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            elif isinstance(embeddings, list):
                return embeddings
            else:
                return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        except Exception as e:
            logger.error(f"Sentence-transformers embedding error: {e}")
            # Return zero vectors as fallback
            try:
                dims = self.model.get_sentence_embedding_dimension()  # Dynamic dimension detection
                if dims is None:
                    dims = 768  # Default for BERT models
            except:
                dims = 768  # Fallback for BERT models
            return [[0.0] * dims] * len(texts)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': f'sentence-transformers-{self.model_name}',
            'provider': 'sentence-transformers',
            'model': self.model_name,
            'device': self.device,
            'dimensions': self.model.get_sentence_embedding_dimension()  # Dynamic dimension detection
        }


class EmbeddingFactory:
    """Factory for creating embedding instances based on configuration"""

    @staticmethod
    def create_embedding(config: Dict[str, Any]) -> EmbeddingInterface:
        """Create embedding instance based on configuration"""
        embedding_type = config.get('type', 'openai')

        if embedding_type == 'openai':
            return OpenAIEmbedding(config)
        elif embedding_type == 'sentence-transformers':
            return SentenceTransformersEmbedding(config)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
