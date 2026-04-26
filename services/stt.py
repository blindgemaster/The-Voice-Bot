"""STT service factory — returns the configured speech-to-text provider."""

from config import Config, STTProvider


# Whisper / Pipecat want ISO-639-1 (`en`, `ur`, ...). Map common 3-letter and
# alternate forms so a friendly value in the UI doesn't crash the pipeline.
_LANGUAGE_ALIASES = {
    "eng": "en", "english": "en",
    "urd": "ur", "urdu": "ur",
    "ara": "ar", "arabic": "ar",
    "hin": "hi", "hindi": "hi",
    "spa": "es", "spanish": "es",
    "fra": "fr", "fre": "fr", "french": "fr",
    "deu": "de", "ger": "de", "german": "de",
    "jpn": "ja", "japanese": "ja",
    "kor": "ko", "korean": "ko",
    "zho": "zh", "chi": "zh", "chinese": "zh",
    "rus": "ru", "russian": "ru",
    "por": "pt", "portuguese": "pt",
    "ita": "it", "italian": "it",
    "tur": "tr", "turkish": "tr",
    "nld": "nl", "dutch": "nl",
}


def _normalize_lang(code: str) -> str:
    code = (code or "").strip().lower()
    return _LANGUAGE_ALIASES.get(code, code)


def create_stt(cfg: Config):
    p = cfg.stt_provider

    if p == STTProvider.WHISPER_CLOUD:
        from pipecat.services.openai.stt import OpenAISTTService, OpenAISTTSettings

        return OpenAISTTService(
            api_key=cfg.openai_api_key,
            language=_normalize_lang(cfg.stt_language),
            settings=OpenAISTTSettings(model="whisper-1"),
        )

    if p == STTProvider.WHISPER_LOCAL:
        from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings, Model

        model_map = {
            "tiny": Model.TINY,
            "base": Model.BASE,
            "small": Model.SMALL,
            "medium": Model.MEDIUM,
            "large": Model.LARGE,
            "large_v3_turbo": Model.LARGE_V3_TURBO,
        }
        model_enum = model_map.get(cfg.local_whisper_model, Model.LARGE_V3_TURBO)
        return WhisperSTTService(
            model=model_enum,
            device="cuda",
            compute_type="float16",
            settings=WhisperSTTSettings(
                language=_normalize_lang(cfg.stt_language),
            ),
        )

    if p == STTProvider.AZURE:
        from pipecat.services.azure.stt import AzureSTTService

        return AzureSTTService(
            api_key=cfg.azure_speech_key,
            region=cfg.azure_speech_region,
            language=_normalize_lang(cfg.stt_language),
        )

    if p == STTProvider.DEEPGRAM:
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(api_key=cfg.deepgram_api_key)

    raise ValueError(f"Unknown STT provider: {p}")
