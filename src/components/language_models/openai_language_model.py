from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
from .base_language_model import BaseLanguageModel


class OpenAILanguageModel(BaseLanguageModel):
    """
    OpenAI Language Model-Implementierung als Baseline.
    
    Nutzt OpenAI's GPT-Modelle für Textgeneration.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, 
                 max_tokens: int = 500, api_key: str = None, **kwargs):
        """
        Initialisiert das OpenAI Language Model.
        
        Args:
            model_name: Name des OpenAI-Modells
            temperature: Temperatur für die Textgeneration
            max_tokens: Maximale Anzahl von Tokens in der Antwort
            api_key: OpenAI API-Schlüssel
            **kwargs: Weitere Parameter
        """
        super().__init__(model_name, temperature, max_tokens, **kwargs)
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
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generiert eine Antwort basierend auf einem Prompt.
        
        Args:
            prompt: Input-Prompt für das Modell
            **kwargs: Zusätzliche Parameter für die Generation
            
        Returns:
            Generierte Antwort als String
        """
        if not prompt.strip():
            raise ValueError("Prompt darf nicht leer sein")
        
        try:
            # Parameter aus kwargs extrahieren oder Standardwerte verwenden
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Fehler bei der Textgeneration: {e}")
    
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
        if not query.strip():
            raise ValueError("Query darf nicht leer sein")
        
        if not context:
            raise ValueError("Kontext darf nicht leer sein")
        
        # System-Prompt aus kwargs extrahieren oder Standard verwenden
        system_prompt = kwargs.pop("system_prompt", None)
        
        # RAG-Prompt erstellen
        rag_prompt = self.create_rag_prompt(query, context, system_prompt)
        
        try:
            # Parameter aus kwargs extrahieren oder Standardwerte verwenden
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Fehler bei der Textgeneration mit Kontext: {e}")
    
    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generiert eine Antwort basierend auf einer Nachrichtenliste.
        
        Args:
            messages: Liste von Nachrichten im OpenAI-Format
            **kwargs: Zusätzliche Parameter für die Generation
            
        Returns:
            Generierte Antwort als String
        """
        if not messages:
            raise ValueError("Nachrichten dürfen nicht leer sein")
        
        try:
            # Parameter aus kwargs extrahieren oder Standardwerte verwenden
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Fehler bei der Textgeneration mit Nachrichten: {e}")
    
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
                "Du bist ein hilfreicher Assistent, der Fragen zur DSGVO (Datenschutz-Grundverordnung) "
                "basierend auf den bereitgestellten Kontextinformationen beantwortet. "
                "Antworte präzise und nur basierend auf dem gegebenen Kontext. "
                "Wenn die Information nicht im Kontext verfügbar ist, sage das explizit. "
                "Gib bei rechtlichen Fragen immer die entsprechenden Artikel an."
            )
        
        # Kontext formatieren
        context_text = "\n\n".join([f"Kontext {i+1}:\n{ctx}" 
                                   for i, ctx in enumerate(context)])
        
        prompt = f"""{system_prompt}

Kontext:
{context_text}

Frage: {query}

Antwort:"""
        
        return prompt
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generiert Antworten für mehrere Prompts.
        
        Args:
            prompts: Liste von Prompts
            **kwargs: Zusätzliche Parameter für die Generation
            
        Returns:
            Liste von generierten Antworten
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                # Bei Fehlern leeren String oder Fehlermeldung hinzufügen
                results.append(f"Fehler bei der Generation: {str(e)}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        model_info = {
            "model_name": self.model_name,
            "provider": "OpenAI",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "supports_streaming": True,
            "supports_functions": True
        }
        
        # Modell-spezifische Informationen
        if "gpt-4" in self.model_name:
            model_info.update({
                "context_window": 128000,
                "training_data_cutoff": "2024-04",
                "capabilities": ["text", "code", "reasoning"]
            })
        elif "gpt-3.5" in self.model_name:
            model_info.update({
                "context_window": 16384,
                "training_data_cutoff": "2021-09",
                "capabilities": ["text", "code"]
            })
        
        return model_info
    
    def estimate_tokens(self, text: str) -> int:
        """
        Schätzt die Anzahl der Tokens in einem Text.
        
        Args:
            text: Text zur Token-Schätzung
            
        Returns:
            Geschätzte Anzahl von Tokens
        """
        # Grobe Schätzung: 1 Token ≈ 4 Zeichen für deutsche Texte
        return len(text) // 4
    
    def check_token_limit(self, text: str) -> bool:
        """
        Prüft, ob ein Text das Token-Limit überschreitet.
        
        Args:
            text: Zu prüfender Text
            
        Returns:
            True wenn Text unter dem Limit liegt
        """
        estimated_tokens = self.estimate_tokens(text)
        
        # Konservative Schätzung der Kontext-Limits
        if "gpt-4" in self.model_name:
            return estimated_tokens < 120000  # Puffer für Antwort
        elif "gpt-3.5" in self.model_name:
            return estimated_tokens < 15000   # Puffer für Antwort
        else:
            return estimated_tokens < 8000    # Konservative Schätzung 