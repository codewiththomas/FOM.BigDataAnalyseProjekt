from typing import List
import numpy as np
from openai import OpenAI
import os
from .base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding-Implementierung als Baseline.
    
    Nutzt OpenAI's text-embedding-3-small Modell für Embeddings.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small", 
                 dimensions: int = 1536, api_key: str = None, **kwargs):
        """
        Initialisiert das OpenAI Embedding.
        
        Args:
            model_name: Name des OpenAI Embedding-Modells
            dimensions: Dimensionalität der Embeddings
            api_key: OpenAI API-Schlüssel
            **kwargs: Weitere Parameter
        """
        super().__init__(model_name, dimensions, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API-Schlüssel ist erforderlich. Setzen Sie OPENAI_API_KEY oder übergeben Sie api_key.")
        
        self.client = None
        self.load_model()
    
    def load_model(self) -> None:
        """
        Lädt das OpenAI Client.
        """
        try:
            self.client = OpenAI(api_key=self.api_key)
            # Test-Aufruf um Verbindung zu prüfen
            self.client.models.list()
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des OpenAI Clients: {e}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Erstellt Embeddings für eine Liste von Texten.
        
        Args:
            texts: Liste von Texten
            
        Returns:
            NumPy-Array mit Embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            # Leere Texte filtern
            filtered_texts = [text.strip() for text in texts if text.strip()]
            
            if not filtered_texts:
                return np.array([])
            
            # OpenAI API-Aufruf
            response = self.client.embeddings.create(
                model=self.model_name,
                input=filtered_texts,
                dimensions=self.dimensions
            )
            
            # Embeddings extrahieren
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Erstellen der Embeddings: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Erstellt Embedding für eine einzelne Query.
        
        Args:
            query: Query-Text
            
        Returns:
            NumPy-Array mit Embedding
        """
        if not query.strip():
            raise ValueError("Query darf nicht leer sein")
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[query.strip()],
                dimensions=self.dimensions
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding)
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Erstellen des Query-Embeddings: {e}")
    
    def batch_embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Erstellt Embeddings für große Textmengen in Batches.
        
        OpenAI hat ein Limit von ~2048 Texten pro Request, daher verwenden wir kleinere Batches.
        
        Args:
            texts: Liste von Texten
            batch_size: Größe der Batches (max 100 für OpenAI)
            
        Returns:
            NumPy-Array mit allen Embeddings
        """
        if not texts:
            return np.array([])
        
        # Batch-Größe für OpenAI anpassen
        batch_size = min(batch_size, 100)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Gibt die Dimensionalität der Embeddings zurück.
        
        Returns:
            Anzahl der Dimensionen
        """
        return self.dimensions
    
    def get_model_info(self) -> dict:
        """
        Gibt Informationen über das Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "provider": "OpenAI",
            "max_input_tokens": 8192,  # Approximation für text-embedding-3-small
            "supported_languages": ["de", "en", "es", "fr", "it", "pt", "zh", "ja", "ko"]
        } 