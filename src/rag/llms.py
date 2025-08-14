from typing import List, Dict, Any
import openai
from .interfaces import LLMInterface
import logging

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API-based LLM implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1000)

        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("No OpenAI API key provided, using environment variable")

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response using OpenAI API"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': f'openai-{self.model}',
            'provider': 'openai',
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


class LocalLLM(LLMInterface):
    """Local LLM implementation (placeholder for on-premises models)"""

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'local-model')
        self.endpoint = config.get('endpoint', 'http://localhost:8000')
        self.api_type = config.get('api_type', 'ollama')  # ollama, vllm, etc.

        logger.info(f"Initialized local LLM: {self.model_name} at {self.endpoint}")

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response using local LLM (placeholder implementation)"""
        # This is a placeholder - you'll need to implement based on your local setup
        logger.warning("Local LLM not fully implemented - returning placeholder response")

        if context:
            return f"Local LLM response to: {prompt[:50]}... (context provided)"
        else:
            return f"Local LLM response to: {prompt[:50]}..."

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': self.model_name,
            'provider': 'local',
            'endpoint': self.endpoint,
            'api_type': self.api_type
        }


class LLMFactory:
    """Factory for creating LLM instances based on configuration"""

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> LLMInterface:
        """Create LLM instance based on configuration"""
        llm_type = config.get('type', 'openai')

        if llm_type == 'openai':
            return OpenAILLM(config)
        elif llm_type == 'local':
            return LocalLLM(config)
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
