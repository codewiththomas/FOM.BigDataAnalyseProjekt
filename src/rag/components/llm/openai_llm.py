"""
Implementiert OpenAI-basierte Language Models für das RAG-System.
"""

import os
from typing import Dict, List, Optional, Union, Any, Generator
from openai import OpenAI

from ...config import config
from .base import LLM


class OpenAILanguageModel(LLM):
    """
    Implementiert ein Language Model mit OpenAI.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        api_key: str = None
    ):
        """
        Initialisiert das OpenAILanguageModel.

        Args:
            model: Das zu verwendende Language Model
            temperature: Die Temperatur für die Generierung (0.0 - 2.0)
            max_tokens: Die maximale Anzahl der zu generierenden Tokens
            api_key: Der OpenAI API-Key
        """
        self.model = model or config.llm.model
        self.temperature = temperature if temperature is not None else config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.api_key = api_key or config.llm.api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generiert eine Antwort basierend auf einem Prompt mit OpenAI.

        Args:
            prompt: Der Prompt für das Language Model
            **kwargs: Weitere Parameter für das Language Model

        Returns:
            Die generierte Antwort als String
        """
        # Zusammenführen von Standard- und benutzerdefinierten Parametern
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        params.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                    {"role": "user", "content": prompt}
                ],
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Fehler bei der Generierung mit OpenAI: {e}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generiert eine Antwort als Stream basierend auf einem Prompt mit OpenAI.

        Args:
            prompt: Der Prompt für das Language Model
            **kwargs: Weitere Parameter für das Language Model

        Returns:
            Ein Generator, der die generierte Antwort stückweise zurückgibt
        """
        # Zusammenführen von Standard- und benutzerdefinierten Parametern
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        params.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                    {"role": "user", "content": prompt}
                ],
                **params
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise ValueError(f"Fehler bei der Stream-Generierung mit OpenAI: {e}")