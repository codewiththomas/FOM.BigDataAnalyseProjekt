"""
Generator module with support for different LLM providers.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os


class BaseGenerator(ABC):
    """Abstract base class for text generators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("name", "unknown")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1000)

    @abstractmethod
    def generate(self, query: str, context: List[str]) -> Dict[str, Any]:
        """
        Generate response using query and context.

        Args:
            query: User question
            context: List of relevant context chunks

        Returns:
            Dictionary with response and metadata
        """
        pass

    def _create_prompt(self, query: str, context: List[str]) -> str:
        """Create prompt from query and context."""
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context)])

        prompt = f"""Du bist ein hilfreicher Assistent, der Fragen basierend auf gegebenem Kontext beantwortet.

Kontext:
{context_text}

Frage: {query}

Antwort: Beantworte die Frage basierend auf dem gegebenen Kontext. Wenn der Kontext nicht ausreicht, um die Frage zu beantworten, sage das deutlich. Antworte auf Deutsch."""

        return prompt

    def _measure_performance(self, func, *args, **kwargs) -> tuple:
        """Measure execution time and tokens per second."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time

        # Estimate tokens (rough approximation)
        response_text = result.get("response", "") if isinstance(result, dict) else str(result)
        estimated_tokens = len(response_text.split()) * 1.3  # Rough token estimation
        tokens_per_second = estimated_tokens / execution_time if execution_time > 0 else 0

        return result, {
            "execution_time": execution_time,
            "estimated_tokens": estimated_tokens,
            "tokens_per_second": tokens_per_second
        }


class OpenAIGenerator(BaseGenerator):
    """OpenAI GPT generator."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def generate(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        prompt = self._create_prompt(query, context)

        def _call_api():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return {
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "cost_estimate": self._estimate_cost(response.usage)
            }

        result, performance = self._measure_performance(_call_api)
        result.update(performance)

        return result

    def _estimate_cost(self, usage) -> float:
        """Estimate cost based on usage (rough estimates)."""
        # Rough cost estimates for GPT-4o-mini (as of 2024)
        cost_per_1k_input = 0.00015  # $0.15 per 1K input tokens
        cost_per_1k_output = 0.0006  # $0.60 per 1K output tokens

        input_cost = (usage.prompt_tokens / 1000) * cost_per_1k_input
        output_cost = (usage.completion_tokens / 1000) * cost_per_1k_output

        return input_cost + output_cost


class GroqGenerator(BaseGenerator):
    """Groq API generator for Mixtral and other models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided")

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("Groq package not installed. Run: pip install groq")

    def generate(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using Groq API."""
        prompt = self._create_prompt(query, context)

        def _call_api():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return {
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        result, performance = self._measure_performance(_call_api)
        result.update(performance)
        result["cost_estimate"] = 0.0  # Groq often has free tier

        return result


class OllamaGenerator(BaseGenerator):
    """Ollama generator for local models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("ollama_endpoint", "http://localhost:11434")

        try:
            import ollama
            self.client = ollama.Client(host=self.endpoint)
        except ImportError:
            raise ImportError("Ollama package not installed. Run: pip install ollama")

    def generate(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using Ollama."""
        prompt = self._create_prompt(query, context)

        def _call_api():
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )

            return {
                "response": response["response"],
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                }
            }

        result, performance = self._measure_performance(_call_api)
        result.update(performance)
        result["cost_estimate"] = 0.0  # Local models have no API cost

        return result


def get_generator(config: Dict[str, Any]) -> BaseGenerator:
    """
    Factory function to create appropriate generator based on config.

    Args:
        config: Configuration dictionary

    Returns:
        Generator instance
    """
    provider = config.get("provider", "").lower()
    model_name = config.get("name", "")

    if provider == "openai" or "gpt" in model_name.lower():
        return OpenAIGenerator(config)
    elif provider == "groq" or "mixtral" in model_name.lower():
        return GroqGenerator(config)
    elif provider == "ollama" or model_name in ["llama-sauerkraut", "qwen2.5-7b", "llama3.1"]:
        return OllamaGenerator(config)
    else:
        raise ValueError(f"Unknown provider: {provider} or model: {model_name}")


# Model configurations for easy switching
AVAILABLE_MODELS = {
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "description": "OpenAI GPT-4o-mini - fast and cost-effective"
    },
    "mixtral-8x7b": {
        "name": "mixtral-8x7b-32768",
        "provider": "groq",
        "description": "Mixtral 8x7B via Groq - fast inference"
    },
    "llama-sauerkraut": {
        "name": "llama3.1:8b-sauerkraut",
        "provider": "ollama",
        "description": "German-tuned Llama model via Ollama"
    },
    "qwen2.5-7b": {
        "name": "qwen2.5:7b-instruct",
        "provider": "ollama",
        "description": "Qwen 2.5 7B Instruct via Ollama"
    }
}


if __name__ == "__main__":
    # Example usage
    sample_config = {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.1,
        "max_tokens": 500,
        "openai_api_key": "your-api-key-here"
    }

    try:
        generator = get_generator(sample_config)

        # Test generation
        query = "Was ist die DSGVO?"
        context = ["Die DSGVO ist eine EU-Verordnung zum Schutz personenbezogener Daten."]

        result = generator.generate(query, context)
        print(f"Response: {result['response']}")
        print(f"Model: {result['model']}")
        print(f"Execution time: {result['execution_time']:.2f}s")

    except Exception as e:
        print(f"Error testing generator: {e}")

    print("Generator test complete!")