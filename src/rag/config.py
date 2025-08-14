import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RAGConfig:
    """Configuration manager for RAG system"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get('llm', {})

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.config.get('embedding', {})

    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        return self.config.get('chunking', {})

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return self.config.get('retrieval', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self.config.get('dataset', {})

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return self.config.get('pipeline', {})

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()

    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_sections = ['llm', 'embedding', 'chunking', 'retrieval']
        missing_sections = [section for section in required_sections
                           if section not in self.config]

        if missing_sections:
            logger.error(f"Missing required configuration sections: {missing_sections}")
            return False

        logger.info("Configuration validation passed")
        return True
