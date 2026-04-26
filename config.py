"""
Central configuration for the Urdu Voice Bot.

`Config` is the single source of truth. Both the CLI (`bot.py`) and the UI
(`ui.py`) call `load_config()` to build a Config from defaults overlaid with
`state.json` (UI-managed settings) and `.env` (API keys), then pass it through
to the service factories and pipeline builder.

Persistence is split:
- `save_state(cfg)` writes provider selection, models, voices, system prompt,
  and VoxCPM tunables to `state.json`. Called automatically on Apply.
- `save_env(cfg)` writes API keys + cloud regions to `.env` and refreshes
  `os.environ`. Called from the UI's "Save to .env" button.

Both files are gitignored.
"""

import json
import os
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv, set_key
from loguru import logger


class STTProvider(Enum):
    WHISPER_CLOUD = "whisper_cloud"
    WHISPER_LOCAL = "whisper_local"
    AZURE = "azure"
    DEEPGRAM = "deepgram"


class LLMProvider(Enum):
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class TTSProvider(Enum):
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    CARTESIA = "cartesia"
    VOXCPM = "voxcpm"


DEFAULT_SYSTEM_PROMPT = (
    "آپ ایک مددگار اردو اسسٹنٹ ہیں۔ "
    "ہمیشہ اردو میں مختصر اور واضح جواب دیں۔ "
    "اگر صارف انگریزی میں بات کرے تو بھی اردو میں جواب دیں۔"
)

ENV_PATH = Path(__file__).parent / ".env"
STATE_PATH = Path(__file__).parent / "state.json"

# Config attribute -> .env key. Only these fields are persisted by save_env().
ENV_FIELDS: dict[str, str] = {
    "openai_api_key":      "OPENAI_API_KEY",
    "azure_speech_key":    "AZURE_SPEECH_KEY",
    "azure_speech_region": "AZURE_SPEECH_REGION",
    "deepgram_api_key":    "DEEPGRAM_API_KEY",
    "anthropic_api_key":   "ANTHROPIC_API_KEY",
    "elevenlabs_api_key":  "ELEVENLABS_API_KEY",
    "elevenlabs_voice_id": "ELEVENLABS_VOICE_ID",
    "cartesia_api_key":    "CARTESIA_API_KEY",
    "cartesia_voice_id":   "CARTESIA_VOICE_ID",
    "ollama_base_url":     "OLLAMA_BASE_URL",
    "lmstudio_base_url":   "LMSTUDIO_BASE_URL",
}

# Fields written to state.json — everything UI-mutable that isn't a secret.
# (Voice IDs are also in ENV_FIELDS so they stay in sync with their API key
# pairs, but state.json is the canonical store for the UI's view of them.)
STATE_FIELDS: tuple[str, ...] = (
    "stt_provider", "llm_provider", "tts_provider",
    "stt_language", "local_whisper_model",
    "ollama_model", "ollama_base_url",
    "lmstudio_model", "lmstudio_base_url",
    "openai_llm_model", "anthropic_llm_model",
    "azure_tts_voice", "openai_tts_voice",
    "elevenlabs_voice_id", "cartesia_voice_id",
    "system_prompt",
    "voxcpm_reference_wav", "voxcpm_cfg_value",
    "voxcpm_inference_timesteps", "voxcpm_output_sample_rate",
)


@dataclass
class Config:
    # Active providers
    stt_provider: STTProvider = STTProvider.WHISPER_LOCAL
    llm_provider: LLMProvider = LLMProvider.LMSTUDIO
    tts_provider: TTSProvider = TTSProvider.AZURE

    # STT
    stt_language: str = "ur"
    local_whisper_model: str = "large_v3_turbo"  # tiny|base|small|medium|large|large_v3_turbo

    # LLM
    ollama_model: str = "aya-expanse:8b"
    ollama_base_url: str = "http://localhost:11434/v1"
    lmstudio_model: str = "google/gemma-3-4b"
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    openai_llm_model: str = "gpt-4o"
    anthropic_llm_model: str = "claude-sonnet-4-20250514"

    # TTS voices
    azure_tts_voice: str = "ur-PK-AsadNeural"
    elevenlabs_voice_id: str = ""
    openai_tts_voice: str = "alloy"
    cartesia_voice_id: str = ""

    # VoxCPM2 (local, openbmb/VoxCPM2). Reference wav is optional — empty means
    # the model's default voice. Other knobs are exposed in case the user wants
    # to trade quality for latency.
    voxcpm_reference_wav: str = ""
    voxcpm_cfg_value: float = 2.0
    # Diffusion steps per chunk. Lower = faster, slightly lower quality.
    # Default of 6 (vs the upstream example's 10) trades quality for real-time
    # feasibility on RTX 30-series GPUs.
    voxcpm_inference_timesteps: int = 6
    voxcpm_output_sample_rate: int = 24000

    # System prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # API keys & cloud regions
    openai_api_key: str = ""
    azure_speech_key: str = ""
    azure_speech_region: str = "eastus2"
    deepgram_api_key: str = ""
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""
    cartesia_api_key: str = ""


def load_config() -> Config:
    """Build a Config: dataclass defaults < state.json < .env / os.environ."""
    cfg = Config()
    _apply_state_file(cfg)
    load_dotenv(ENV_PATH, override=False)
    for attr, env_key in ENV_FIELDS.items():
        val = os.getenv(env_key)
        if val:
            setattr(cfg, attr, val)
    return cfg


def save_env(cfg: Config) -> None:
    """Write env-backed fields from cfg to .env and refresh os.environ."""
    ENV_PATH.touch(exist_ok=True)
    for attr, env_key in ENV_FIELDS.items():
        val = getattr(cfg, attr) or ""
        set_key(str(ENV_PATH), env_key, val, quote_mode="never")
        os.environ[env_key] = val


def save_state(cfg: Config) -> None:
    """Write UI-managed fields from cfg to state.json (atomic)."""
    out: dict = {}
    for attr in STATE_FIELDS:
        val = getattr(cfg, attr)
        if isinstance(val, Enum):
            val = val.value
        out[attr] = val
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _apply_state_file(cfg: Config) -> None:
    """Overlay state.json onto cfg in-place. No-op if file missing/unreadable."""
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"state.json unreadable, falling back to defaults: {e}")
        return

    field_types = {f.name: f.type for f in fields(cfg)}
    for attr in STATE_FIELDS:
        if attr not in data:
            continue
        val = data[attr]
        current = getattr(cfg, attr)
        if isinstance(current, Enum):
            try:
                val = type(current)(val)
            except (ValueError, TypeError):
                continue  # stale enum value — keep the default
        setattr(cfg, attr, val)
