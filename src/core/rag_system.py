from typing import List, Dict, Any
from components.chunkers import LineChunker
from components.embeddings import SentenceTransformersEmbedding
from components.vector_stores import InMemoryVectorStore
from components.language_models import OpenAILanguageModel
from config import RAGConfig

class RAGSystem:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunker = LineChunker()
        self.embedding = SentenceTransformersEmbedding()
        self.vector_store = InMemoryVectorStore()
        
        # Language Model mit API-Key aus Konfiguration initialisieren
        api_key = config.language_model_params.get("api_key")
        self.language_model = OpenAILanguageModel(api_key=api_key)
        
    def process_documents(self, documents: List[str]) -> None:
        """Verarbeitet Dokumente und speichert sie im Vector Store"""
        all_chunks = []
        all_embeddings = []
        
        for i, doc in enumerate(documents):
            # Chunking
            chunks = self.chunker.split_document(doc)
            all_chunks.extend(chunks)
            
            # Embeddings erstellen
            embeddings = self.embedding.embed_texts(chunks)
            all_embeddings.extend(embeddings)
        
        # Im Vector Store speichern
        self.vector_store.add_texts(all_chunks, all_embeddings)
        print(f"Verarbeitet: {len(documents)} Dokumente -> {len(all_chunks)} Chunks")
    
    def query(self, question: str, k: int = 4) -> str:
        """Beantwortet eine Frage basierend auf den gespeicherten Dokumenten"""
        # Ähnliche Dokumente finden
        similar_docs = self.vector_store.similarity_search(question, k)
        
        # Kontext extrahieren
        context = [doc["text"] for doc in similar_docs]
        
        # Antwort generieren
        response = self.language_model.generate_response(question, context)
        
        return response 