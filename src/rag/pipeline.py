from typing import List, Dict, Any, Optional
import time
import logging
from interfaces import (
    LLMInterface, EmbeddingInterface, ChunkingInterface,
    RetrievalInterface, Chunk, QueryResult
)
from cache import RAGCache

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components"""

    def __init__(self,
                 llm: LLMInterface,
                 embedding: EmbeddingInterface,
                 chunking: ChunkingInterface,
                 retrieval: RetrievalInterface,
                 cache: Optional[RAGCache] = None):
        self.llm = llm
        self.embedding = embedding
        self.chunking = chunking
        self.retrieval = retrieval

        #if hasattr(self.retrieval, "set_embedding_model"):
        self.retrieval.set_embedding_model(self.embedding)

        self.cache = cache
        self.is_indexed = False

        logger.info("RAG Pipeline initialized")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents by chunking, embedding, and adding to retrieval"""
        logger.info(f"Indexing {len(documents)} documents...")

        # Dokumente vor Chunking speichern
        self.documents_before_chunking = documents

        # Step 1: Chunking (with caching)
        chunks = self._get_or_create_chunks(documents)

        # Step 2: Embeddings (with caching)
        embeddings = self._get_or_create_embeddings(chunks)

        # Step 3: Add to retrieval
        self.retrieval.add_chunks(chunks, embeddings)

        self.is_indexed = True
        logger.info(f"Indexed {len(chunks)} chunks")

    def _get_or_create_chunks(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Get chunks from cache or create new ones"""
        if self.cache:
            # Try to load from cache
            chunking_config = {
                'type': self.chunking.__class__.__name__,
                'chunk_size': getattr(self.chunking, 'chunk_size', None),
                'chunk_overlap': getattr(self.chunking, 'chunk_overlap', None),
                'separator': getattr(self.chunking, 'separator', None)
            }

            cached_chunks = self.cache.load_chunks(chunking_config)
            if cached_chunks:
                logger.info("Using cached chunks")
                return cached_chunks

        # Create new chunks
        logger.info("Creating new chunks...")
        chunks = self.chunking.chunk_documents(documents)

        # Save to cache if available
        if self.cache:
            chunking_config = {
                'type': self.chunking.__class__.__name__,
                'chunk_size': getattr(self.chunking, 'chunk_size', None),
                'chunk_overlap': getattr(self.chunking, 'chunk_overlap', None),
                'separator': getattr(self.chunking, 'separator', None)
            }
            self.cache.save_chunks(chunks, chunking_config)

        return chunks

    def _get_or_create_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Get embeddings from cache or create new ones"""
        if self.cache:
            # Try to load from cache - only use serializable values
            embedding_config = {}

            # Get model name safely
            if hasattr(self.embedding, 'model_name'):
                embedding_config['model_name'] = self.embedding.model_name
            elif hasattr(self.embedding, 'model'):
                # For OpenAI, get the model string
                model_obj = self.embedding.model
                if hasattr(model_obj, 'name'):
                    embedding_config['model'] = model_obj.name
                else:
                    embedding_config['model'] = str(model_obj)
            else:
                embedding_config['model'] = 'unknown'

            # Get device safely
            if hasattr(self.embedding, 'device'):
                embedding_config['device'] = str(self.embedding.device)

            # Add type
            embedding_config['type'] = self.embedding.__class__.__name__

            cached_embeddings = self.cache.load_embeddings(embedding_config)
            if cached_embeddings:
                logger.info("Using cached embeddings")
                return cached_embeddings

        # Create new embeddings
        logger.info("Creating new embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding.embed(texts)

        # Save to cache if available
        if self.cache:
            # Use the same config structure
            embedding_config = {}

            # Get model name safely
            if hasattr(self.embedding, 'model_name'):
                embedding_config['model_name'] = self.embedding.model_name
            elif hasattr(self.embedding, 'model'):
                # For OpenAI, get the model string
                model_obj = self.embedding.model
                if hasattr(model_obj, 'name'):
                    embedding_config['model'] = model_obj.name
                else:
                    embedding_config['model'] = str(model_obj)
            else:
                embedding_config['model'] = 'unknown'

            # Get device safely
            if hasattr(self.embedding, 'device'):
                embedding_config['device'] = str(self.embedding.device)

            # Add type
            embedding_config['type'] = self.embedding.__class__.__name__

            self.cache.save_embeddings(embeddings, embedding_config)

        return embeddings

    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """Process a query and return results"""
        if not self.is_indexed:
            raise RuntimeError("Documents must be indexed before querying")

        start_time = time.time()

        # Retrieve relevant chunks
        retrieved_chunks = self.retrieval.retrieve(question, top_k)

        # Generate response
        context = "\n\n".join([chunk.text for chunk in retrieved_chunks])
        response = self.llm.generate(question, context)

        query_time = time.time() - start_time

        return QueryResult(
            query=question,
            chunks=retrieved_chunks,
            response=response,
            metadata={
                'query_time': query_time,
                'chunks_retrieved': len(retrieved_chunks),
                'llm_model': self.llm.get_model_info().get('name', 'unknown'),
                'embedding_model': self.embedding.get_model_info().get('name', 'unknown'),
                'retrieval_method': self.retrieval.get_model_info().get('name', 'unknown')
            }
        )

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            'llm': self.llm.get_model_info(),
            'embedding': self.embedding.get_model_info(),
            'chunking': {
                'name': f"{self.chunking.__class__.__name__.lower()}-chunking",
                'strategy': getattr(self.chunking, 'chunk_size', 'variable'),
                'chunk_size': getattr(self.chunking, 'chunk_size', None),
                'chunk_overlap': getattr(self.chunking, 'chunk_overlap', None),
                'separator': getattr(self.chunking, 'separator', None)
            },
            'retrieval': self.retrieval.get_model_info(),
            'is_indexed': self.is_indexed,
            'dataset': {
                'documents_processed': len(self.documents_before_chunking) if hasattr(self,
                                                                                      'documents_before_chunking') else 'unknown',
                'grouping_applied': any(
                    doc.get('metadata', {}).get('grouped', False) for doc in self.documents_before_chunking) if hasattr(
                    self, 'documents_before_chunking') else False
            }
        }
