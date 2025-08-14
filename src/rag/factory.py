from typing import Dict, Any
from config import RAGConfig
from llms import LLMFactory
from embeddings import EmbeddingFactory
from chunking import ChunkingFactory
from retrieval import RetrievalFactory
from pipeline import RAGPipeline
from evaluation import EvaluationManager
from cache import RAGCache
import logging

logger = logging.getLogger(__name__)


class RAGFactory:
    """Factory for creating complete RAG systems from configuration"""

    def __init__(self, config_path: str):
        self.config = RAGConfig(config_path)
        self.config.validate_config()

        # Extract experiment name from config path
        self.experiment_name = self._extract_experiment_name(config_path)

        # Initialize cache
        self.cache = RAGCache(self.experiment_name)

        logger.info(f"RAG Factory initialized for experiment: {self.experiment_name}")

    def _extract_experiment_name(self, config_path: str) -> str:
        """Extract experiment name from config file path"""
        from pathlib import Path
        config_file = Path(config_path)
        return config_file.stem  # Remove .yaml extension

    def create_pipeline(self) -> RAGPipeline:
        """Create a complete RAG pipeline from configuration"""
        logger.info("Creating RAG pipeline from configuration...")

        # Create components
        llm = LLMFactory.create_llm(self.config.get_llm_config())
        embedding = EmbeddingFactory.create_embedding(self.config.get_embedding_config())
        chunking = ChunkingFactory.create_chunking(self.config.get_chunking_config())
        retrieval = RetrievalFactory.create_retrieval(self.config.get_retrieval_config())

        # Create pipeline with cache
        pipeline = RAGPipeline(
            llm=llm,
            embedding=embedding,
            chunking=chunking,
            retrieval=retrieval,
            cache=self.cache
        )

        logger.info("RAG pipeline created successfully")
        return pipeline

    def create_evaluator(self) -> EvaluationManager:
        """Create evaluation manager from configuration"""
        logger.info("Creating evaluation manager from configuration...")

        evaluation_config = self.config.get_evaluation_config()
        enabled_metrics = evaluation_config.get('enabled_metrics', [])

        evaluators = []

        if 'precision-recall' in enabled_metrics:
            from evaluation import PrecisionRecallEvaluator
            evaluators.append(PrecisionRecallEvaluator())

        if 'timing' in enabled_metrics:
            from evaluation import TimingEvaluator
            evaluators.append(TimingEvaluator())

        if 'ragas' in enabled_metrics:
            from evaluation import RAGASEvaluator
            evaluators.append(RAGASEvaluator())

        evaluator = EvaluationManager(evaluators)
        logger.info(f"Evaluation manager created with {len(evaluators)} evaluators")

        return evaluator

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        return self.cache.get_cache_info()

    def clear_cache(self) -> None:
        """Clear the cache for this experiment"""
        self.cache.clear_cache()
