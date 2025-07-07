from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLanguageModel(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generiert Antwort basierend auf Query und Kontext"""
        pass 