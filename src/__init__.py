"""
ResearchRAG - Modulares RAG-System für wissenschaftliche Studien

Ein flexibles und erweiterbares RAG-System, das es ermöglicht, verschiedene
Komponenten (Chunker, Embeddings, Vector Stores, Language Models) einfach
auszutauschen und zu vergleichen.

Hauptkomponenten:
- config: Konfigurationssystem
- core: RAG-Pipeline und Component Loader
- components: Austauschbare RAG-Komponenten
- evaluations: Evaluierungsmetriken
- utils: Hilfsfunktionen

Autoren: FOM Research Team
Projekt: Big Data Analyse - RAG-System Evaluierung
"""

__version__ = "0.1.0"
__author__ = "FOM Research Team"
__email__ = "research@fom.de"

from .core.rag_pipeline import RAGPipeline
from .core.component_loader import ComponentLoader
from .config.pipeline_configs import PipelineConfig, get_baseline_config

__all__ = [
    "RAGPipeline",
    "ComponentLoader", 
    "PipelineConfig",
    "get_baseline_config"
] 