from typing import List, Dict, Any
import openai
import requests
import json
import logging
from interfaces import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API-based LLM implementation"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1000)

        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            logger.warning("No OpenAI API key provided, using environment variable")
            self.client = openai.OpenAI()

        logger.info(f"Initialized OpenAI LLM: {self.model}")

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response using OpenAI API"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information, say so."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except openai.APIError as e:
            logger.error(f"OpenAI LLM error: {e}")
            return f"Error generating response: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI LLM: {e}")
            return f"Unexpected error: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': f'openai-{self.model}',
            'provider': 'openai',
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


class LocalLLM(LLMInterface):
    """Local LLM implementation using Ollama"""

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'llama3:8b')
        self.endpoint = config.get('endpoint', 'http://localhost:11434')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1000)
        self.api_type = config.get('api_type', 'ollama')

        # Test connection
        self._test_connection()

        logger.info(f"Initialized local LLM: {self.model_name} at {self.endpoint}")

    def _test_connection(self):
        """Test connection to the local LLM service"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to {self.endpoint}")
            else:
                logger.warning(f"Connection test failed with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to {self.endpoint}: {e}")
            logger.warning("Local LLM may not be available")

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response using local LLM"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            if self.api_type == 'ollama':
                return self._generate_ollama(full_prompt)
            else:
                logger.warning(f"Unknown API type: {self.api_type}, falling back to Ollama")
                return self._generate_ollama(full_prompt)

        except Exception as e:
            logger.error(f"Error in local LLM generation: {e}")
            return f"Error generating response: {e}"

    def _generate_ollama(self, prompt: str) -> str:
        """Generate response using Ollama API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }

            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"API error: {response.status_code}"

        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out")
            return "Request timed out"
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return f"Request failed: {e}"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return "Invalid response format"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': self.model_name,
            'provider': 'local',
            'endpoint': self.endpoint,
            'api_type': self.api_type,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
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
