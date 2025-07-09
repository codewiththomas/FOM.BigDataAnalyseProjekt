import json
import requests
import time
from typing import Dict, Any, Optional, List
from .base_language_model import BaseLanguageModel


class OllamaLanguageModel(BaseLanguageModel):
    """
    Ollama language model implementation for local LLM inference.

    This class interfaces with Ollama to run language models locally
    without requiring API keys or external services.
    """

    def __init__(self,
                 model_name: str = "llama3.2",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 timeout: int = 60,
                 system_prompt: Optional[str] = None):
        """
        Initialize Ollama language model.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt
        """
        super().__init__()

        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.system_prompt = system_prompt

        # Check if Ollama is available
        self.available = self._check_ollama_availability()

        if self.available:
            # Check if model is available
            self._ensure_model_available()

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama service is running")
                return True
            else:
                print(f"Ollama service responded with status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Ollama service not available: {e}")
            print("Make sure Ollama is installed and running: https://ollama.ai/")
            return False

    def _ensure_model_available(self):
        """Ensure the specified model is available."""
        try:
            # Get list of available models
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]

                if self.model_name not in model_names:
                    print(f"Model '{self.model_name}' not found. Available models: {model_names}")
                    print(f"Pulling model '{self.model_name}'...")
                    self._pull_model()
                else:
                    print(f"Model '{self.model_name}' is available")
            else:
                print(f"Failed to get model list: {response.status_code}")
        except Exception as e:
            print(f"Error checking model availability: {e}")

    def _pull_model(self):
        """Pull the specified model from Ollama."""
        try:
            pull_data = {"name": self.model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=pull_data,
                timeout=300  # 5 minutes timeout for model pulling
            )

            if response.status_code == 200:
                print(f"Model '{self.model_name}' pulled successfully")
            else:
                print(f"Failed to pull model '{self.model_name}': {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error pulling model: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Ollama model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self.available:
            raise RuntimeError("Ollama service not available")

        # Merge kwargs with default parameters
        generation_params = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k)
        }

        try:
            # Prepare request data
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": generation_params["temperature"],
                    "num_predict": generation_params["max_tokens"],
                    "top_p": generation_params["top_p"],
                    "top_k": generation_params["top_k"]
                }
            }

            # Add system prompt if provided
            if self.system_prompt:
                request_data["system"] = self.system_prompt

            # Make request
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")

                # Store generation info
                self._last_generation_info = {
                    "model": self.model_name,
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                    "generation_time": end_time - start_time,
                    "parameters": generation_params
                }

                return generated_text
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error generating text with Ollama: {e}")
            raise

    def generate_with_context(self,
                            context: str,
                            question: str,
                            **kwargs) -> str:
        """
        Generate answer based on context and question.

        Args:
            context: Context information
            question: Question to answer
            **kwargs: Additional generation parameters

        Returns:
            Generated answer
        """
        # Create a RAG prompt template
        rag_prompt = f"""Kontext: {context}

Frage: {question}

Antwort: Basierend auf dem gegebenen Kontext, beantworte die Frage präzise und sachlich. Wenn die Antwort nicht im Kontext zu finden ist, sage das deutlich."""

        return self.generate(rag_prompt, **kwargs)

    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.

        Returns:
            List of available model names
        """
        if not self.available:
            return []

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            else:
                print(f"Failed to get model list: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "timeout": self.timeout,
            "available": self.available,
            "model_type": "ollama"
        }

        if self.available:
            try:
                # Get model details
                response = requests.post(
                    f"{self.base_url}/api/show",
                    json={"name": self.model_name},
                    timeout=10
                )

                if response.status_code == 200:
                    model_details = response.json()
                    info.update({
                        "model_details": model_details.get("details", {}),
                        "model_info": model_details.get("modelinfo", {}),
                        "parameters": model_details.get("parameters", {})
                    })
            except Exception as e:
                info["error"] = f"Failed to get model details: {e}"

        return info

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for the generation (always 0 for local models).

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost (always 0 for local models)
        """
        return 0.0

    def get_generation_info(self) -> Dict[str, Any]:
        """
        Get information about the last generation.

        Returns:
            Dictionary containing generation information
        """
        return getattr(self, '_last_generation_info', {})

    def stream_generate(self, prompt: str, **kwargs):
        """
        Generate text with streaming response.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        if not self.available:
            raise RuntimeError("Ollama service not available")

        # Merge kwargs with default parameters
        generation_params = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k)
        }

        try:
            # Prepare request data
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": generation_params["temperature"],
                    "num_predict": generation_params["max_tokens"],
                    "top_p": generation_params["top_p"],
                    "top_k": generation_params["top_k"]
                }
            }

            # Add system prompt if provided
            if self.system_prompt:
                request_data["system"] = self.system_prompt

            # Make streaming request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                stream=True,
                timeout=self.timeout
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                yield chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error streaming text with Ollama: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text (if model supports it).

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        if not self.available:
            raise RuntimeError("Ollama service not available")

        try:
            request_data = {
                "model": self.model_name,
                "prompt": text
            }

            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=request_data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                raise Exception(f"Ollama embeddings API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error generating embeddings with Ollama: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the Ollama service.

        Returns:
            Health check results
        """
        health_info = {
            "service_available": False,
            "model_available": False,
            "response_time": None,
            "error": None
        }

        try:
            # Check service availability
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            end_time = time.time()

            if response.status_code == 200:
                health_info["service_available"] = True
                health_info["response_time"] = end_time - start_time

                # Check if specific model is available
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                health_info["model_available"] = self.model_name in model_names
                health_info["available_models"] = model_names
            else:
                health_info["error"] = f"Service responded with status: {response.status_code}"

        except Exception as e:
            health_info["error"] = str(e)

        return health_info

    def set_system_prompt(self, system_prompt: str):
        """
        Set system prompt for the model.

        Args:
            system_prompt: System prompt to set
        """
        self.system_prompt = system_prompt
        print(f"System prompt set for model {self.model_name}")

    def clear_system_prompt(self):
        """Clear the system prompt."""
        self.system_prompt = None
        print(f"System prompt cleared for model {self.model_name}")

    def benchmark_model(self, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Benchmark the model performance on test prompts.

        Args:
            test_prompts: List of prompts to benchmark on

        Returns:
            Dictionary with benchmark results
        """
        if not self.available:
            return {"error": "Ollama service not available"}

        results = []
        total_time = 0
        total_tokens = 0

        for prompt in test_prompts:
            try:
                start_time = time.time()
                response = self.generate(prompt)
                end_time = time.time()

                generation_time = end_time - start_time
                token_count = len(response.split())

                results.append({
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "generation_time": generation_time,
                    "tokens_generated": token_count,
                    "tokens_per_second": token_count / generation_time if generation_time > 0 else 0
                })

                total_time += generation_time
                total_tokens += token_count

            except Exception as e:
                results.append({
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "error": str(e)
                })

        return {
            "model_name": self.model_name,
            "num_prompts": len(test_prompts),
            "total_time": total_time,
            "total_tokens": total_tokens,
            "avg_time_per_prompt": total_time / len(test_prompts) if test_prompts else 0,
            "avg_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "results": results
        }