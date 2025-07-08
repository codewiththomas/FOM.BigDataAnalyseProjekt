from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from pathlib import Path
from ..config.pipeline_configs import PipelineConfig
from .component_loader import ComponentLoader
from ..components.chunkers import BaseChunker
from ..components.embeddings import BaseEmbedding
from ..components.vector_stores import BaseVectorStore
from ..components.language_models import BaseLanguageModel


class RAGPipeline:
    """
    Hauptklasse für die RAG-Pipeline.

    Orchestriert alle Komponenten und stellt die Hauptfunktionalitäten bereit.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialisiert die RAG-Pipeline.

        Args:
            config: Pipeline-Konfiguration
        """
        self.config = config
        self.component_loader = ComponentLoader()

        # Komponenten laden
        self.chunker = self._load_chunker()
        self.embedding = self._load_embedding()
        self.vector_store = self._load_vector_store()
        self.language_model = self._load_language_model()

        # Status
        self.is_indexed = False
        self.indexed_document_count = 0

        # Logging
        self.logger = logging.getLogger(__name__)

    def _load_chunker(self) -> BaseChunker:
        """Lädt den Chunker basierend auf der Konfiguration."""
        chunker_config = self.config.get_chunker_config()
        return self.component_loader.load_chunker(chunker_config)

    def _load_embedding(self) -> BaseEmbedding:
        """Lädt das Embedding-Modell basierend auf der Konfiguration."""
        embedding_config = self.config.get_embedding_config()
        return self.component_loader.load_embedding(embedding_config)

    def _load_vector_store(self) -> BaseVectorStore:
        """Lädt den Vector Store basierend auf der Konfiguration."""
        vector_store_config = self.config.get_vector_store_config()
        return self.component_loader.load_vector_store(vector_store_config)

    def _load_language_model(self) -> BaseLanguageModel:
        """Lädt das Language Model basierend auf der Konfiguration."""
        llm_config = self.config.get_language_model_config()
        return self.component_loader.load_language_model(llm_config)

    def load_documents_from_file(self, file_path: str) -> List[str]:
        """
        Lädt Dokumente aus einer Datei.

        Args:
            file_path: Pfad zur Datei

        Returns:
            Liste von Dokumenten
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Für DSGVO-Datei: Behandle als ein großes Dokument
            return [content]

        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der Datei '{file_path}': {e}")

    def index_documents(self, documents: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Indexiert Dokumente in der Pipeline.

        Args:
            documents: Liste von Dokumenten
            show_progress: Ob Fortschritt angezeigt werden soll

        Returns:
            Dictionary mit Indexierungs-Statistiken
        """
        start_time = time.time()

        if show_progress:
            print(f"Indexiere {len(documents)} Dokument(e)...")

        # 1. Chunking
        if show_progress:
            print("1. Chunking...")

        all_chunks = []
        for doc_id, document in enumerate(documents):
            chunks = self.chunker.chunk_text(document)

            for chunk_id, chunk in enumerate(chunks):
                chunk_data = {
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "metadata": {
                        "source_doc": doc_id,
                        "chunk_index": chunk_id,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk)
                    }
                }
                all_chunks.append(chunk_data)

        if show_progress:
            print(f"   → {len(all_chunks)} Chunks erstellt")

        # 2. Embedding
        if show_progress:
            print("2. Embedding-Erstellung...")

        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding.batch_embed_texts(chunk_texts)

        if show_progress:
            print(f"   → {len(embeddings)} Embeddings erstellt")

        # 3. Vector Store
        if show_progress:
            print("3. Vector Store-Indexierung...")

        metadata_list = [chunk["metadata"] for chunk in all_chunks]
        chunk_ids = self.vector_store.add_texts(chunk_texts, embeddings, metadata_list)

        if show_progress:
            print(f"   → {len(chunk_ids)} Chunks im Vector Store gespeichert")

        # Status aktualisieren
        self.is_indexed = True
        self.indexed_document_count = len(documents)

        # Statistiken
        end_time = time.time()
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "total_embeddings": len(embeddings),
            "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "indexing_time": end_time - start_time,
            "chunks_per_second": len(all_chunks) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            "average_chunk_length": sum(len(chunk["text"]) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        }

        if show_progress:
            print(f"✓ Indexierung abgeschlossen in {stats['indexing_time']:.2f}s")
            print(f"  Durchschnittliche Chunk-Länge: {stats['average_chunk_length']:.0f} Zeichen")
            print(f"  Chunks pro Sekunde: {stats['chunks_per_second']:.1f}")

        return stats

    def query(self, question: str, top_k: Optional[int] = None,
              return_context: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Führt eine Query gegen die Pipeline aus.

        Args:
            question: Frage/Query
            top_k: Anzahl der zu retrievenden Dokumente (None = aus Konfiguration)
            return_context: Ob der Kontext zurückgegeben werden soll
            **kwargs: Zusätzliche Parameter für das Language Model

        Returns:
            Dictionary mit Antwort und Metadaten
        """
        if not self.is_indexed:
            raise RuntimeError("Pipeline ist nicht indexiert. Rufen Sie zuerst index_documents() auf.")

        start_time = time.time()

        # Top-K aus Konfiguration oder Parameter
        if top_k is None:
            top_k = self.config.get_vector_store_config().get("top_k", 5)

        # 1. Query Embedding
        query_embedding = self.embedding.embed_query(question)

        # 2. Retrieval
        retrieval_results = self.vector_store.similarity_search(query_embedding, top_k)

        # 3. Kontext für Generation vorbereiten
        context_texts = [result["text"] for result in retrieval_results]

        # 4. Generation
        answer = self.language_model.generate_with_context(question, context_texts, **kwargs)

        # Timing
        end_time = time.time()

        # Ergebnis zusammenstellen
        result = {
            "question": question,
            "answer": answer,
            "retrieval_count": len(retrieval_results),
            "query_time": end_time - start_time,
            "pipeline_config": self.config.get_component_types()
        }

        if return_context:
            result["retrieved_contexts"] = retrieval_results

        return result

    def batch_query(self, questions: List[str], top_k: Optional[int] = None,
                   show_progress: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        Führt mehrere Queries in einem Batch aus.

        Args:
            questions: Liste von Fragen
            top_k: Anzahl der zu retrievenden Dokumente
            show_progress: Ob Fortschritt angezeigt werden soll
            **kwargs: Zusätzliche Parameter für das Language Model

        Returns:
            Liste von Query-Ergebnissen
        """
        if not self.is_indexed:
            raise RuntimeError("Pipeline ist nicht indexiert. Rufen Sie zuerst index_documents() auf.")

        results = []

        if show_progress:
            print(f"Verarbeite {len(questions)} Fragen...")

        for i, question in enumerate(questions):
            if show_progress and i % 10 == 0:
                print(f"  Fortschritt: {i+1}/{len(questions)}")

            try:
                result = self.query(question, top_k=top_k, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Fehler bei Frage {i+1}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Fehler: {str(e)}",
                    "error": True
                })

        if show_progress:
            print("✓ Batch-Verarbeitung abgeschlossen")

        return results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über die Pipeline zurück.

        Returns:
            Dictionary mit Pipeline-Informationen
        """
        return {
            "pipeline_name": self.config.get_pipeline_info().get("name", "Unknown"),
            "pipeline_version": self.config.get_pipeline_info().get("version", "Unknown"),
            "components": self.config.get_component_types(),
            "is_indexed": self.is_indexed,
            "indexed_documents": self.indexed_document_count,
            "vector_store_stats": self.vector_store.get_stats() if self.is_indexed else None
        }

    def get_component_info(self) -> Dict[str, Any]:
        """
        Gibt detaillierte Informationen über alle Komponenten zurück.

        Returns:
            Dictionary mit Komponenten-Informationen
        """
        info = {
            "chunker": {
                "type": self.chunker.__class__.__name__,
                "config": self.chunker.get_config()
            },
            "embedding": {
                "type": self.embedding.__class__.__name__,
                "config": self.embedding.get_config()
            },
            "vector_store": {
                "type": self.vector_store.__class__.__name__,
                "config": self.vector_store.get_config(),
                "stats": self.vector_store.get_stats() if self.is_indexed else None
            },
            "language_model": {
                "type": self.language_model.__class__.__name__,
                "config": self.language_model.get_config()
            }
        }

        return info

    def clear_index(self) -> bool:
        """
        Löscht den Index im Vector Store.

        Returns:
            True wenn erfolgreich gelöscht
        """
        success = self.vector_store.clear()
        if success:
            self.is_indexed = False
            self.indexed_document_count = 0
        return success

    def save_pipeline_config(self, file_path: str) -> None:
        """
        Speichert die Pipeline-Konfiguration in eine Datei.

        Args:
            file_path: Pfad zur Datei
        """
        self.config.save_to_file(file_path)

    @classmethod
    def load_from_config_file(cls, config_file: str) -> 'RAGPipeline':
        """
        Lädt eine Pipeline aus einer Konfigurationsdatei.

        Args:
            config_file: Pfad zur Konfigurationsdatei

        Returns:
            RAG-Pipeline-Instanz
        """
        config = PipelineConfig.load_from_file(config_file)
        return cls(config)

    def __str__(self) -> str:
        return f"RAGPipeline(config={self.config.get_pipeline_info().get('name', 'Unknown')})"

    def __repr__(self) -> str:
        return self.__str__()