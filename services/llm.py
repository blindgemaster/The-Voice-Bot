"""LLM service factory — returns the configured language model provider."""

import config
from config import LLMProvider


def create_llm():
    provider = config.LLM

    if provider == LLMProvider.OLLAMA:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.base_llm import OpenAILLMSettings

        return OpenAILLMService(
            api_key="ollama",
            base_url=config.OLLAMA_BASE_URL,
            settings=OpenAILLMSettings(model=config.OLLAMA_MODEL),
        )

    if provider == LLMProvider.LMSTUDIO:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.base_llm import OpenAILLMSettings

        return OpenAILLMService(
            api_key="lm-studio",
            base_url=config.LMSTUDIO_BASE_URL,
            settings=OpenAILLMSettings(model=config.LMSTUDIO_MODEL),
        )

    if provider == LLMProvider.OPENAI:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.base_llm import OpenAILLMSettings

        return OpenAILLMService(
            api_key=config.OPENAI_API_KEY,
            settings=OpenAILLMSettings(model=config.OPENAI_LLM_MODEL),
        )

    if provider == LLMProvider.ANTHROPIC:
        from pipecat.services.anthropic import AnthropicLLMService

        return AnthropicLLMService(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.ANTHROPIC_LLM_MODEL,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
