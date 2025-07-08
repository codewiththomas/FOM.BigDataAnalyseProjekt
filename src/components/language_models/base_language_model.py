from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class BaseLanguageModel(ABC):
    """
    Abstrakte Basisklasse für alle Language Model-Implementierungen.
    
    Diese Klasse definiert das Interface für verschiedene Language Model-Implementierungen
    und stellt gemeinsame Funktionalitäten bereit.
    """
    
    def __init__(self, model_name: str, temperature: float = 0.1, 
                 max_tokens: int = 500, **kwargs):
        """
        Initialisiert das Language Model.
        
        Args:
            model_name: Name des Language Models
            temperature: Temperatur für die Textgeneration
            max_tokens: Maximale Anzahl von Tokens in der Antwort
            **kwargs: Weitere modell-spezifische Parameter
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs
        self._model = None
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generiert eine Antwort basierend auf einem Prompt.
        
        Args:
            prompt: Input-Prompt für das Modell
            **kwargs: Zusätzliche Parameter für die Generation
            
        Returns:
            Generierte Antwort als String
        """
        pass
    
    @abstractmethod
    def generate_with_context(self, query: str, context: List[str], **kwargs) -> str:
        """
        Generiert eine Antwort mit gegebenem Kontext.
        
        Args:
            query: Benutzer-Query
            context: Liste von Kontext-Dokumenten
            **kwargs: Zusätzliche Parameter für die Generation
            
        Returns:
            Generierte Antwort als String
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Lädt das Language Model.
        """
        pass
    
    def create_rag_prompt(self, query: str, context: List[str], 
                         system_prompt: Optional[str] = None) -> str:
        """
        Erstellt einen RAG-Prompt aus Query und Kontext.
        
        Args:
            query: Benutzer-Query
            context: Liste von Kontext-Dokumenten
            system_prompt: Optionaler System-Prompt
            
        Returns:
            Formatierter Prompt
        """
        if system_prompt is None:
            system_prompt = (
                "Du bist ein hilfreicher Assistent, der Fragen basierend auf "
                "den bereitgestellten Kontextinformationen beantwortet. "
                "Antworte nur basierend auf dem gegebenen Kontext und gib an, "
                "wenn die Information nicht verfügbar ist."
            )
        
        context_text = "\n\n".join([f"Kontext {i+1}:\n{ctx}" 
                                   for i, ctx in enumerate(context)])
        
        prompt = f"""{system_prompt}

Kontext:
{context_text}

Frage: {query}

Antwort:"""
        
        return prompt
    
    def validate_response(self, response: str, query: str) -> Dict[str, Any]:
        """
        Validiert eine generierte Antwort.
        
        Args:
            response: Generierte Antwort
            query: Ursprüngliche Query
            
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "response_length": len(response),
            "word_count": len(response.split())
        }
        
        # Grundlegende Validierungen
        if not response.strip():
            validation["is_valid"] = False
            validation["issues"].append("Leere Antwort")
        
        if len(response) > self.max_tokens * 4:  # Grober Schätzwert
            validation["issues"].append("Antwort zu lang")
        
        if "Ich kann diese Frage nicht beantworten" in response:
            validation["issues"].append("Modell konnte nicht antworten")
        
        return validation
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generiert Antworten für mehrere Prompts.
        
        Args:
            prompts: Liste von Prompts
            **kwargs: Zusätzliche Parameter für die Generation
            
        Returns:
            Liste von generierten Antworten
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration des Language Models zurück.
        
        Returns:
            Dictionary mit Konfigurationsparametern
        """
        return {
            "type": self.__class__.__name__,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.config
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, temp={self.temperature})"
    
    def __repr__(self) -> str:
        return self.__str__() 