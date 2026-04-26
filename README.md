# The Voice Bot

An Urdu speech-to-speech voice bot built on [Pipecat](https://github.com/pipecat-ai/pipecat). Speak in Urdu (or English), and the bot replies in spoken Urdu. Every stage of the pipeline — speech-to-text, LLM, text-to-speech — is a swappable provider, so you can mix local (GPU) inference with cloud APIs depending on your latency, cost, and privacy needs.

## Architecture

```
mic → STT → LLM → TTS → speakers
```

The pipeline is built once in [pipeline.py](pipeline.py) and is transport-agnostic. The transport (local audio today, WebRTC later) is injected at startup so the same pipeline can drive a desktop session or a browser session.

### Supported providers

| Stage | Providers |
|-------|-----------|
| STT   | OpenAI Whisper (cloud), faster-whisper (local GPU), Azure, Deepgram |
| LLM   | Ollama, LM Studio, OpenAI, Anthropic |
| TTS   | Azure, ElevenLabs, OpenAI, Cartesia |

Default stack is local-first: faster-whisper (`large_v3_turbo`) on CUDA → LM Studio (`gemma-3-4b`) → Azure Neural TTS (`ur-PK-AsadNeural`).

## Requirements

- Python 3.11–3.13 (3.12 recommended; 3.14 is too new for some ML deps)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A working microphone and speakers
- For the local-GPU stack: an NVIDIA GPU with CUDA support (tested on RTX 3060, 12GB VRAM)
- For local LLMs: [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai/) running on `localhost`
- API keys for whichever cloud providers you enable

## Setup

```bash
git clone https://github.com/<your-user>/the-voice-bot.git
cd the-voice-bot
uv sync
```

Copy `.env.example` to `.env` and fill in only the keys you need:

```bash
cp .env.example .env
```

```ini
# STT
OPENAI_API_KEY=
DEEPGRAM_API_KEY=

# TTS
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=eastus2
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
CARTESIA_API_KEY=
CARTESIA_VOICE_ID=

# LLM
OLLAMA_BASE_URL=http://localhost:11434/v1
ANTHROPIC_API_KEY=
```

### Optional extras

The default install includes the providers used by the default stack. To enable others:

```bash
uv sync --extra deepgram        # Deepgram STT
uv sync --extra elevenlabs      # ElevenLabs TTS
uv sync --extra cartesia        # Cartesia TTS
uv sync --extra anthropic       # Anthropic LLM
uv sync --extra webrtc          # WebRTC transport (for the upcoming web UI)
uv sync --extra all             # everything
```

## Run

```bash
uv run python bot.py
```

You should see `Listening... speak in Urdu or English`. Speak into the mic; the bot replies through your speakers. Voice activity detection (Silero VAD) handles turn-taking, and interruptions are enabled.

## Configuration

All provider selection and model/voice settings live in [config.py](config.py). To change the stack, edit the three lines near the top:

```python
STT = STTProvider.WHISPER_LOCAL
LLM = LLMProvider.LMSTUDIO
TTS = TTSProvider.AZURE
```

Other things you can tweak in `config.py`:
- `STT_LANGUAGE` — defaults to `ur` (Urdu)
- `LOCAL_WHISPER_MODEL` — `tiny` / `base` / `small` / `medium` / `large` / `large_v3_turbo`
- `OLLAMA_MODEL`, `LMSTUDIO_MODEL`, `OPENAI_LLM_MODEL`, `ANTHROPIC_LLM_MODEL`
- `AZURE_TTS_VOICE` — `ur-PK-AsadNeural` (male) or `ur-PK-UzmaNeural` (female)
- `SYSTEM_PROMPT` — the assistant's persona, written in Urdu

## Project layout

```
.
├── bot.py              # entrypoint — wires transport into the pipeline
├── pipeline.py         # transport-agnostic Pipecat pipeline
├── config.py           # provider selection, models, voices, system prompt
├── services/
│   ├── stt.py          # STT provider factory
│   ├── llm.py          # LLM provider factory
│   └── tts.py          # TTS provider factory
└── transports/
    ├── local.py        # local mic + speakers (PyAudio)
    └── webrtc.py       # SmallWebRTCTransport stub for the web UI
```

## Roadmap

- [ ] Browser UI over `SmallWebRTCTransport` (no Daily.co dependency)
- [ ] Conversation logging / transcript export
- [ ] Per-turn metrics (STT/LLM/TTS latency)

## License

Not yet specified — add a `LICENSE` file before publishing if you want the code to be reusable.
