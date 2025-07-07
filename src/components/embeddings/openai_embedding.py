from typing import List, Dict, Any, Optional
from .base_embedding import BaseEmbedding
import openai
import numpy as np


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding Wrapper für die Verwendung von OpenAI's Embedding-Modellen.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model

        if api_key:
            openai.api_key = api_key

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Erstellt Embeddings für eine Liste von Texten.
        """
        if not self.api_key:
            raise ValueError("OpenAI API Key ist erforderlich")

        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model
            )

            # Extrahiere Embeddings aus der Antwort
            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            raise Exception(f"Fehler beim Erstellen der OpenAI Embeddings: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        """
        Erstellt ein Embedding für einen einzelnen Text.
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Gibt die Dimension der Embeddings zurück.
        """
        # OpenAI text-embedding-ada-002 hat 1536 Dimensionen
        if self.model == "text-embedding-ada-002":
            return 1536
        elif self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            # Fallback
            return 1536

    def get_config(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "embedding_dimension": self.get_embedding_dimension()
        }