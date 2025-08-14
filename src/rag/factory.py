from typing import Dict, Any
from config import RAGConfig
from llms import LLMFactory
from embeddings import EmbeddingFactory
from chunking import ChunkingFactory
from retrieval import RetrievalFactory
from pipeline import RAGPipeline
from evaluation import EvaluationManager
import logging

logger = logging.getLogger(__name__)


class RAGFactory:
    """Factory for creating complete RAG systems from configuration"""

    def __init__(self, config_path: str):
        self.config = RAGConfig(config_path)
        self.config.validate_config()

    def create_pipeline(self) -> RAGPipeline:
        """Create a complete RAG pipeline from configuration"""
        logger.info("Creating RAG pipeline from configuration...")

        # Create components
        llm = LLMFactory.create_llm(self.config.get_llm_config())
        embedding = EmbeddingFactory.create_embedding(self.config.get_embedding_config())
        chunking = ChunkingFactory.create_chunking(self.config.get_chunking_config())
        retrieval = RetrievalFactory.create_retrieval(self.config.get_retrieval_config())

        # Create pipeline
        pipeline = RAGPipeline(llm, embedding, chunking, retrieval)

        logger.info("RAG pipeline created successfully")
        return pipeline

    def create_evaluator(self) -> EvaluationManager:
        """Create evaluation manager from configuration"""
        logger.info("Creating evaluation manager...")

        evaluator = EvaluationManager(self.config.get_all_config())

        logger.info("Evaluation manager created successfully")
        return evaluator

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the current configuration"""
        return {
            'config_path': str(self.config.config_path),
            'pipeline_info': self.config.get_all_config(),
            'evaluation_info': self.config.get_evaluation_config()
        }
