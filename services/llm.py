"""LLM service factory — returns the configured language model provider."""

from config import Config, LLMProvider


def create_llm(cfg: Config):
    p = cfg.llm_provider

    if p == LLMProvider.OLLAMA:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.base_llm import OpenAILLMSettings

        return OpenAILLMService(
            api_key="ollama",
            base_url=cfg.ollama_base_url,
            settings=OpenAILLMSettings(model=cfg.ollama_model),
        )

    if p == LLMProvider.LMSTUDIO:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.base_llm import OpenAILLMSettings

        return OpenAILLMService(
            api_key="lm-studio",
            base_url=cfg.lmstudio_base_url,
            settings=OpenAILLMSettings(model=cfg.lmstudio_model),
        )

    if p == LLMProvider.OPENAI:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.base_llm import OpenAILLMSettings

        return OpenAILLMService(
            api_key=cfg.openai_api_key,
            settings=OpenAILLMSettings(model=cfg.openai_llm_model),
        )

    if p == LLMProvider.ANTHROPIC:
        from pipecat.services.anthropic import AnthropicLLMService

        return AnthropicLLMService(
            api_key=cfg.anthropic_api_key,
            model=cfg.anthropic_llm_model,
        )

    raise ValueError(f"Unknown LLM provider: {p}")
