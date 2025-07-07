from typing import List, Dict, Any, Optional
from .base_language_model import BaseLanguageModel
import requests
import json


class LocalLanguageModel(BaseLanguageModel):
    """
    Lokaler Language Model Wrapper für die Verwendung von Small Language Models.
    Unterstützt verschiedene lokale Modelle über API-Endpunkte.
    """

    def __init__(self, model_name: str = "llama2", api_url: str = "http://localhost:11434", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_url = api_url
        self.api_endpoint = f"{api_url}/api/generate"

    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generiert eine Antwort basierend auf dem gegebenen Prompt.
        """
        try:
            # Payload für die API-Anfrage
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            # API-Anfrage senden
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                raise Exception(f"API-Fehler: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Netzwerkfehler: {str(e)}")
        except Exception as e:
            raise Exception(f"Fehler bei der Antwortgenerierung: {str(e)}")

    def generate_chat_response(self, messages: List[Dict[str, str]],
                             max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generiert eine Antwort für ein Chat-Format.
        """
        # Konvertiere Messages zu einem Prompt
        prompt = self._messages_to_prompt(messages)
        return self.generate_response(prompt, max_tokens, temperature)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Konvertiert Chat-Messages zu einem einzelnen Prompt.
        """
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant: "
        return prompt

    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das verwendete Modell zurück.
        """
        return {
            "model_name": self.model_name,
            "api_url": self.api_url,
            "model_type": "local"
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_url": self.api_url
        }