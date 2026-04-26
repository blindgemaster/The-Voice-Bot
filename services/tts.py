"""TTS service factory — returns the configured text-to-speech provider."""

from config import Config, TTSProvider


def create_tts(cfg: Config):
    p = cfg.tts_provider

    if p == TTSProvider.AZURE:
        from pipecat.services.azure.tts import AzureTTSService, AzureTTSSettings

        return AzureTTSService(
            api_key=cfg.azure_speech_key,
            region=cfg.azure_speech_region,
            settings=AzureTTSSettings(voice=cfg.azure_tts_voice),
        )

    if p == TTSProvider.ELEVENLABS:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        return ElevenLabsTTSService(
            api_key=cfg.elevenlabs_api_key,
            voice_id=cfg.elevenlabs_voice_id,
        )

    if p == TTSProvider.OPENAI:
        from pipecat.services.openai.tts import OpenAITTSService

        return OpenAITTSService(
            api_key=cfg.openai_api_key,
            voice=cfg.openai_tts_voice,
        )

    if p == TTSProvider.CARTESIA:
        from pipecat.services.cartesia.tts import CartesiaTTSService

        return CartesiaTTSService(
            api_key=cfg.cartesia_api_key,
            voice_id=cfg.cartesia_voice_id,
        )

    if p == TTSProvider.VOXCPM:
        try:
            from services.voxcpm_tts import VoxCPMTTSService
        except ImportError as e:
            raise ImportError(
                "VoxCPM is not installed. Run: uv sync --extra voxcpm"
            ) from e
        return VoxCPMTTSService(
            reference_wav=cfg.voxcpm_reference_wav,
            cfg_value=cfg.voxcpm_cfg_value,
            inference_timesteps=cfg.voxcpm_inference_timesteps,
            sample_rate=cfg.voxcpm_output_sample_rate,
        )

    raise ValueError(f"Unknown TTS provider: {p}")
