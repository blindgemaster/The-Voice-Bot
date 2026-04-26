"""STT service factory — returns the configured speech-to-text provider."""

import config
from config import STTProvider


def create_stt():
    provider = config.STT

    if provider == STTProvider.WHISPER_CLOUD:
        from pipecat.services.openai.stt import OpenAISTTService, OpenAISTTSettings

        return OpenAISTTService(
            api_key=config.OPENAI_API_KEY,
            language=config.STT_LANGUAGE,
            settings=OpenAISTTSettings(model="whisper-1"),
        )

    if provider == STTProvider.WHISPER_LOCAL:
        from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings, Model

        model_map = {
            "tiny": Model.TINY,
            "base": Model.BASE,
            "small": Model.SMALL,
            "medium": Model.MEDIUM,
            "large": Model.LARGE,
            "large_v3_turbo": Model.LARGE_V3_TURBO,
        }
        model_enum = model_map.get(config.LOCAL_WHISPER_MODEL, Model.LARGE_V3_TURBO)
        return WhisperSTTService(
            model=model_enum,
            device="cuda",
            compute_type="float16",
            settings=WhisperSTTSettings(
                language=config.STT_LANGUAGE,
            ),
        )

    if provider == STTProvider.AZURE:
        from pipecat.services.azure.stt import AzureSTTService

        return AzureSTTService(
            api_key=config.AZURE_SPEECH_KEY,
            region=config.AZURE_SPEECH_REGION,
            language=config.STT_LANGUAGE,
        )

    if provider == STTProvider.DEEPGRAM:
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=config.DEEPGRAM_API_KEY,
        )

    raise ValueError(f"Unknown STT provider: {provider}")
