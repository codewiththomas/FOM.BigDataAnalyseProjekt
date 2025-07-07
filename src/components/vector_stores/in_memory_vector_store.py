from .base_vector_store import BaseVectorStore
from typing import List, Dict, Any
import numpy as np

class InMemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self.texts = []
        self.embeddings = []
        self.metadatas = []
    
    def add_texts(self, texts: List[str], embeddings: List[List[float]], 
                  metadatas: List[Dict[str, Any]] = None):
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        
        # Einfache Implementierung - in der Praxis würden Sie hier
        # das Query-Embedding berechnen
        query_embedding = [0.1] * len(self.embeddings[0])  # Placeholder
        
        # Einfache Ähnlichkeitsberechnung
        similarities = []
        for embedding in self.embeddings:
            similarity = sum(a * b for a, b in zip(query_embedding, embedding))
            similarities.append(similarity)
        
        # Top-k Ergebnisse
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "similarity": similarities[idx]
            })
        
        return results 