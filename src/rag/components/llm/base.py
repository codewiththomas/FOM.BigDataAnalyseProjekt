"""
Basisklassen und Interfaces für Language Models im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Generator


class LLM(ABC):
    """
    Basisklasse für alle Language Models.

    Ermöglicht die Interaktion mit verschiedenen Language Models.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generiert eine Antwort basierend auf einem Prompt.

        Args:
            prompt: Der Prompt für das Language Model
            **kwargs: Weitere Parameter für das Language Model

        Returns:
            Die generierte Antwort als String
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generiert eine Antwort als Stream basierend auf einem Prompt.

        Args:
            prompt: Der Prompt für das Language Model
            **kwargs: Weitere Parameter für das Language Model

        Returns:
            Ein Generator, der die generierte Antwort stückweise zurückgibt
        """
        pass

    def format_prompt(self, query: str, context: str) -> str:
        """
        Formatiert einen Prompt für das RAG-System.

        Args:
            query: Die Benutzeranfrage
            context: Der Kontext aus dem Retrieval-Schritt

        Returns:
            Ein formatierter Prompt für das Language Model
        """
        return f"""Du bist ein Assistent, der Fragen zur DSGVO (Datenschutz-Grundverordnung) beantwortet.
Beantworte die folgende Frage ausschließlich basierend auf dem gegebenen Kontext.

Wenn der Kontext die Information "Es wurden keine relevanten Dokumente gefunden" enthält oder die Antwort nicht ausreichend im Kontext enthalten ist, antworte mit:
"Ich kann diese Frage nicht beantworten, da die nötigen Informationen nicht im bereitgestellten Kontext enthalten sind."

Erfinde KEINE Informationen. Nutze KEIN zusätzliches Wissen, das nicht im Kontext enthalten ist. Wenn du dir unsicher bist, sage, dass du die Frage nicht beantworten kannst.

Kontext:
{context}

Frage:
{query}

Antwort (basiere deine Antwort ausschließlich auf dem bereitgestellten Kontext):"""