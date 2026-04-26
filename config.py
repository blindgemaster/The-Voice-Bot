"""
Central configuration for the Urdu Voice Bot.
Change providers here — the rest of the codebase reads from this file.
"""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Provider enums
# ---------------------------------------------------------------------------

class STTProvider(Enum):
    WHISPER_CLOUD = "whisper_cloud"      # OpenAI Whisper API
    WHISPER_LOCAL = "whisper_local"       # faster-whisper on GPU
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


# ---------------------------------------------------------------------------
# Active providers — change these to swap services
# ---------------------------------------------------------------------------

STT = STTProvider.WHISPER_LOCAL
LLM = LLMProvider.LMSTUDIO
TTS = TTSProvider.AZURE


# ---------------------------------------------------------------------------
# Model / voice settings
# ---------------------------------------------------------------------------

# STT
STT_LANGUAGE = "ur"  # Urdu

# LLM
OLLAMA_MODEL = "aya-expanse:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LMSTUDIO_MODEL = "google/gemma-3-4b"
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_LLM_MODEL = "gpt-4o"
ANTHROPIC_LLM_MODEL = "claude-sonnet-4-20250514"

# TTS
AZURE_TTS_VOICE = "ur-PK-AsadNeural"      # male  (alt: ur-PK-UzmaNeural)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
OPENAI_TTS_VOICE = "alloy"
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")

# Local Whisper model (tiny | base | small | medium | large | large_v3_turbo)
LOCAL_WHISPER_MODEL = "large_v3_turbo"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "آپ ایک مددگار اردو اسسٹنٹ ہیں۔ "
    "ہمیشہ اردو میں مختصر اور واضح جواب دیں۔ "
    "اگر صارف انگریزی میں بات کرے تو بھی اردو میں جواب دیں۔"
    # You are a helpful Urdu assistant.
    # Always respond briefly and clearly in Urdu.
    # Even if the user speaks English, respond in Urdu.
)


# ---------------------------------------------------------------------------
# API keys (read from .env)
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus2")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
