from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

######################################################################################################################
# Definiert, welche Klassen unser RAG besitzen muss und welche Methoden die einzelnen Klassen jeweils haben müssen
######################################################################################################################

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class QueryResult:
    """Represents a query result with retrieved chunks"""
    query: str
    chunks: List[Chunk]
    response: str
    metadata: Dict[str, Any]


class LLMInterface(ABC):
    """Abstract interface for language models"""

    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response from prompt and optional context"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        pass


class EmbeddingInterface(ABC):
    """Abstract interface for embedding models"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        pass


class ChunkingInterface(ABC):
    """Abstract interface for text chunking strategies"""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks with metadata"""
        pass

    @abstractmethod
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get chunking strategy information"""
        pass


class RetrievalInterface(ABC):
    """Abstract interface for retrieval methods"""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the retrieval index"""
        pass

    @abstractmethod
    def set_embedding_model(self, embedding_model: EmbeddingInterface) -> None:
        """Legt fest, welches Embedding genutzt werden soll"""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve top-k relevant chunks for a query"""
        pass

    @abstractmethod
    def get_retrieval_info(self) -> Dict[str, Any]:
        """Get retrieval method information"""
        pass


class EvaluationInterface(ABC):
    """Abstract interface for evaluation metrics"""

    @abstractmethod
    def evaluate(self, query: str, ground_truth: str, response: str,
                retrieved_chunks: List[Chunk]) -> Dict[str, float]:
        """Evaluate a single query-response pair"""
        pass

    @abstractmethod
    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information and description"""
        pass
