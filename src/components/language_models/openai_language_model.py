from .base_language_model import BaseLanguageModel
from typing import List
import openai

class OpenAILanguageModel(BaseLanguageModel):
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.model = model
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            print("⚠️  Kein OpenAI API Key gesetzt - verwende Platzhalter-Antworten")
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generiert Antwort basierend auf Query und Kontext"""
        context_text = "\n".join(context)
        prompt = f"""Kontext:\n{context_text}\n\nFrage: {query}\n\nAntwort basierend auf dem Kontext:"""
        
        try:
            # Prüfe ob OpenAI API Key gesetzt ist
            if not self.client:
                return f"Antwort auf '{query}' basierend auf {len(context)} Kontext-Dokumenten (OpenAI API Key nicht gesetzt)"
            
            # Echte OpenAI API Anfrage (openai>=1.0.0)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Assistent. Antworte basierend auf dem gegebenen Kontext."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Fehler bei der Antwortgenerierung: {e}" 