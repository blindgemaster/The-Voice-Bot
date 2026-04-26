"""TTS service factory — returns the configured text-to-speech provider."""

import config
from config import TTSProvider


def create_tts():
    provider = config.TTS

    if provider == TTSProvider.AZURE:
        from pipecat.services.azure.tts import AzureTTSService, AzureTTSSettings

        return AzureTTSService(
            api_key=config.AZURE_SPEECH_KEY,
            region=config.AZURE_SPEECH_REGION,
            settings=AzureTTSSettings(voice=config.AZURE_TTS_VOICE),
        )

    if provider == TTSProvider.ELEVENLABS:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        return ElevenLabsTTSService(
            api_key=config.ELEVENLABS_API_KEY,
            voice_id=config.ELEVENLABS_VOICE_ID,
        )

    if provider == TTSProvider.OPENAI:
        from pipecat.services.openai.tts import OpenAITTSService

        return OpenAITTSService(
            api_key=config.OPENAI_API_KEY,
            voice=config.OPENAI_TTS_VOICE,
        )

    if provider == TTSProvider.CARTESIA:
        from pipecat.services.cartesia.tts import CartesiaTTSService

        return CartesiaTTSService(
            api_key=config.CARTESIA_API_KEY,
            voice_id=config.CARTESIA_VOICE_ID,
        )

    raise ValueError(f"Unknown TTS provider: {provider}")
