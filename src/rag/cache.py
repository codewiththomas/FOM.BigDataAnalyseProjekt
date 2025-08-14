import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class RAGCache:
    """Caching system for RAG components to save tokens and computation"""

    def __init__(self, experiment_name: str, cache_dir: str = "cache"):
        self.experiment_name = experiment_name
        self.cache_dir = Path(cache_dir) / experiment_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized cache for experiment: {experiment_name}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _get_cache_path(self, component: str, suffix: str = "") -> Path:
        """Get cache file path for a component"""
        filename = f"{self.experiment_name}_{component}{suffix}"
        return self.cache_dir / filename

    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash from configuration to detect changes"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def save_embeddings(self, embeddings: List[List[float]], config: Dict[str, Any]) -> None:
        """Save embeddings to cache"""
        config_hash = self._get_config_hash(config)
        cache_path = self._get_cache_path("embeddings", f"_{config_hash}.pkl")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved {len(embeddings)} embeddings to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to cache: {e}")

    def load_embeddings(self, config: Dict[str, Any]) -> Optional[List[List[float]]]:
        """Load embeddings from cache if available"""
        config_hash = self._get_config_hash(config)
        cache_path = self._get_cache_path("embeddings", f"_{config_hash}.pkl")

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded {len(embeddings)} embeddings from cache: {cache_path}")
                return embeddings
            except Exception as e:
                logger.error(f"Failed to load embeddings from cache: {e}")

        return None

    def save_chunks(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Save chunks to cache"""
        config_hash = self._get_config_hash(config)
        cache_path = self._get_cache_path("chunks", f"_{config_hash}.pkl")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"Saved {len(chunks)} chunks to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save chunks to cache: {e}")

    def load_chunks(self, config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Load chunks from cache if available"""
        config_hash = self._get_config_hash(config)
        cache_path = self._get_cache_path("chunks", f"_{config_hash}.pkl")

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    chunks = pickle.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from cache: {cache_path}")
                return chunks
            except Exception as e:
                logger.error(f"Failed to load chunks from cache: {e}")

        return None

    def save_evaluation_results(self, results: List[Dict[str, Any]], suffix: str = "") -> None:
        """Save evaluation results to cache"""
        cache_path = self._get_cache_path("evaluation_results", f"{suffix}.json")

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results to cache: {e}")

    def load_evaluation_results(self, suffix: str = "") -> Optional[List[Dict[str, Any]]]:
        """Load evaluation results from cache if available"""
        cache_path = self._get_cache_path("evaluation_results", f"{suffix}.json")

        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Loaded evaluation results from cache: {cache_path}")
                return results
            except Exception as e:
                logger.error(f"Failed to load evaluation results from cache: {e}")

        return None

    def clear_cache(self) -> None:
        """Clear all cached files for this experiment"""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared cache for experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached files"""
        cache_info = {
            'experiment_name': self.experiment_name,
            'cache_dir': str(self.cache_dir),
            'cached_files': []
        }

        if self.cache_dir.exists():
            for file_path in self.cache_dir.glob("*"):
                cache_info['cached_files'].append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size if file_path.is_file() else 0,
                    'type': 'file' if file_path.is_file() else 'directory'
                })

        return cache_info
