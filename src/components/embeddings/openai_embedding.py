from typing import List, Dict, Any, Optional
try:
    import openai
except ImportError:
    openai = None
from .base_embedding import BaseEmbedding
import numpy as np


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embeddings - Baseline Implementation

    Verwendet OpenAI's text-embedding-ada-002 Modell für hochwertige Embeddings.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """
        Initialisiert OpenAI Embeddings.

        Args:
            api_key: OpenAI API Key
            model: Embedding-Modell (default: text-embedding-ada-002)
        """
        if openai is None:
            raise ImportError("OpenAI package nicht installiert. Installieren Sie es mit: pip install openai")

        self.api_key = api_key
        self.model = model

        # OpenAI Client initialisieren
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        """
        Erstellt Embedding für einen einzelnen Text.

        Args:
            text: Text für Embedding

        Returns:
            Embedding-Vektor als Liste von Floats
        """
        if not text or not text.strip():
            return [0.0] * 1536  # Standard-Dimensionalität für ada-002

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip()
            )
            return response.data[0].embedding

        except Exception as e:
            print(f"⚠️ OpenAI Embedding Fehler: {e}")
            # Fallback: Null-Vektor
            return [0.0] * 1536

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Erstellt Embeddings für mehrere Texte (Batch-Verarbeitung).

        Args:
            texts: Liste von Texten

        Returns:
            Liste von Embedding-Vektoren
        """
        if not texts:
            return []

        # Leere Texte filtern
        valid_texts = [text.strip() for text in texts if text and text.strip()]

        if not valid_texts:
            return [[0.0] * 1536] * len(texts)

        try:
            # Batch-Request an OpenAI
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts
            )

            # Embeddings extrahieren
            embeddings = [data.embedding for data in response.data]

            # Für leere Texte Null-Vektoren einfügen
            result = []
            valid_idx = 0

            for original_text in texts:
                if original_text and original_text.strip():
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result.append([0.0] * 1536)

            return result

        except Exception as e:
            print(f"⚠️ OpenAI Batch Embedding Fehler: {e}")
            # Fallback: Einzelne Requests
            return [self.embed_text(text) for text in texts]

    def get_embedding_dimension(self) -> int:
        """
        Gibt die Dimensionalität der Embeddings zurück.

        Returns:
            Embedding-Dimensionalität
        """
        if self.model == "text-embedding-ada-002":
            return 1536
        elif self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            return 1536  # Standard-Fallback

    def get_config(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "embedding_dimension": self.get_embedding_dimension()
        }