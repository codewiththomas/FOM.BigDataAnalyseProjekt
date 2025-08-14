from typing import List, Dict, Any, Optional
import time
import logging
from .interfaces import (
    LLMInterface, EmbeddingInterface, ChunkingInterface,
    RetrievalInterface, Chunk, QueryResult
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components"""

    def __init__(self,
                 llm: LLMInterface,
                 embedding: EmbeddingInterface,
                 chunking: ChunkingInterface,
                 retrieval: RetrievalInterface):
        self.llm = llm
        self.embedding = embedding
        self.chunking = chunking
        self.retrieval = retrieval
        self.is_indexed = False

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents by chunking, embedding, and adding to retrieval"""
        logger.info(f"Indexing {len(documents)} documents...")

        all_chunks = []
        for doc in documents:
            # Chunk the document
            chunks = self.chunking.chunk(doc['text'], doc.get('metadata', {}))

            # Generate embeddings for chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding.embed(texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                all_chunks.append(chunk)

        # Add chunks to retrieval system
        self.retrieval.add_chunks(all_chunks)
        self.is_indexed = True
        logger.info(f"Indexed {len(all_chunks)} chunks")

    def query(self, query: str, top_k: int = 5,
              include_context: bool = True) -> QueryResult:
        """Process a query through the RAG pipeline"""
        if not self.is_indexed:
            raise RuntimeError("Documents must be indexed before querying")

        start_time = time.time()

        # Retrieve relevant chunks
        retrieved_chunks = self.retrieval.retrieve(query, top_k)

        # Generate response
        if include_context and retrieved_chunks:
            context = "\n\n".join([chunk.text for chunk in retrieved_chunks])
            prompt = f"""Based on the following context, answer the question.
            If the context doesn't contain enough information, say so.

            Context:
            {context}

            Question: {query}

            Answer:"""
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        response = self.llm.generate(prompt)

        # Calculate timing
        end_time = time.time()
        query_time = end_time - start_time

        metadata = {
            'query_time': query_time,
            'chunks_retrieved': len(retrieved_chunks),
            'llm_model': self.llm.get_model_info().get('name', 'unknown'),
            'embedding_model': self.embedding.get_model_info().get('name', 'unknown'),
            'retrieval_method': self.retrieval.get_retrieval_info().get('name', 'unknown')
        }

        return QueryResult(
            query=query,
            chunks=retrieved_chunks,
            response=response,
            metadata=metadata
        )

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about all pipeline components"""
        return {
            'llm': self.llm.get_model_info(),
            'embedding': self.embedding.get_model_info(),
            'chunking': self.chunking.get_chunking_info(),
            'retrieval': self.retrieval.get_retrieval_info(),
            'is_indexed': self.is_indexed
        }
