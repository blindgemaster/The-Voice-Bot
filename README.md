# The Voice Bot

A speech-to-speech voice bot built on [Pipecat](https://github.com/pipecat-ai/pipecat), with a desktop UI for swapping STT, LLM, and TTS providers live mid-conversation. Every stage is generic — it works in English or anything Whisper can transcribe.

The headline feature: pick any combination of providers — local-GPU Whisper, OpenAI, Anthropic, Ollama, LM Studio, Azure, ElevenLabs, Cartesia, or local VoxCPM2 — change them mid-call, and the pipeline rebuilds without losing conversation history.

## Architecture

```
mic → STT → LLM → TTS → speakers
```

The pipeline is built once in [pipeline.py](pipeline.py) and is transport-agnostic. The Tk UI runs on the main thread; a [controller.py](controller.py) worker thread owns the asyncio loop that drives the Pipecat pipeline. Communication is via thread-safe queues — UI commands flow one way, status / transcript / error events flow the other.

When you swap providers in the UI, the controller tears the live `PipelineTask` down, builds a new one with the updated config, and seeds it with the existing message history so the LLM keeps its memory across the swap.

### Supported providers

| Stage | Providers |
|-------|-----------|
| STT   | OpenAI Whisper (cloud), faster-whisper (local GPU), Azure, Deepgram |
| LLM   | Ollama, LM Studio, OpenAI, Anthropic |
| TTS   | Azure, ElevenLabs, OpenAI, Cartesia, VoxCPM2 (local GPU) |

Default stack is local-first: faster-whisper (`large_v3_turbo`) on CUDA → LM Studio (`gemma-3-4b`) → Azure Neural TTS (`ur-PK-AsadNeural`).

## Requirements

- Python 3.11–3.13 (3.12 recommended; 3.14 is too new for some ML deps)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A working microphone and speakers
- For the local-GPU stack: an NVIDIA GPU with CUDA support (tested on RTX 3060, 12 GB VRAM)
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

You can also paste keys into the UI and click `Save to .env` — the values are written back to this file and to the live `os.environ` so the next pipeline rebuild uses them without restarting the app.

### Optional extras

The default install covers the providers used by the default stack. Other providers are opt-in:

```bash
uv sync --extra deepgram        # Deepgram STT
uv sync --extra elevenlabs      # ElevenLabs TTS
uv sync --extra cartesia        # Cartesia TTS
uv sync --extra anthropic       # Anthropic LLM
uv sync --extra webrtc          # WebRTC transport (for an eventual web UI)
uv sync --extra voxcpm          # VoxCPM2 local TTS (see notes below)
uv sync --extra all             # everything except voxcpm
```

The `voxcpm` extra is heavier than the others — it pulls a CUDA build of PyTorch (~2.6 GB), `triton-windows` for `torch.compile`, `scipy` for resampling, and the `voxcpm` package itself. It also redirects the project's torch/torchaudio install to `https://download.pytorch.org/whl/cu128`, so the CUDA libs match. First run downloads the 5 GB VoxCPM2 weights from Hugging Face into `~/.cache/huggingface`.

## Run

### UI (recommended)

```bash
uv run python ui.py
```

A window opens with:
- **Providers** — three dropdowns (STT / LLM / TTS) with a model field for each, plus a language code field. Change anything → click `Apply` → pipeline rebuilds in 1–3 s.
- **API keys & endpoints** — masked entry fields for every supported provider key, plus base URLs for Ollama and LM Studio. `Save to .env` persists; `Apply` picks up unsaved edits for the current session.
- **System prompt** — editable textarea. The current default is in Urdu; replace with anything.
- **Transcript** — appended as turns commit (in v1, transcript display is best-effort and may lag).
- **Start / Stop** — bring the pipeline up or down. The status pill (top right) shows `idle / starting / running / rebuilding / error`.

### CLI

The original headless mode still works:

```bash
uv run python bot.py
```

It loads the same `Config` from `.env` plus defaults and runs the pipeline once until you Ctrl+C.

## Configuration

The single source of truth is the `Config` dataclass in [config.py](config.py). `load_config()` builds it from defaults overlaid with `.env`; `save_env(cfg)` writes the env-backed fields back. The UI mutates a copy and passes it to the controller; the CLI uses defaults straight from `.env`.

Things you can tweak:
- `stt_provider`, `llm_provider`, `tts_provider` — the active stack
- `stt_language` — defaults to `ur` (3-letter codes like `eng`/`urd` are auto-normalized)
- `local_whisper_model` — `tiny | base | small | medium | large | large_v3_turbo`
- `ollama_model`, `lmstudio_model`, `openai_llm_model`, `anthropic_llm_model`
- `azure_tts_voice` — `ur-PK-AsadNeural` (male) / `ur-PK-UzmaNeural` (female), or any other Azure neural voice
- `voxcpm_inference_timesteps` — diffusion steps per chunk; lower is faster, lower quality (default `6`)
- `voxcpm_reference_wav` — optional path to a reference WAV for voice cloning
- `system_prompt` — the assistant's persona

## Notes on VoxCPM2

VoxCPM2 is a 2 B-parameter local diffusion TTS — fully offline once weights are downloaded. There are a few real tradeoffs to know:

- **VRAM**: the model needs ~8 GB. On a 12 GB card you can pair it with local Whisper, but a local LLM (LM Studio etc.) on the same card will OOM. Use a cloud LLM if you want both local STT and local TTS.
- **Real-time on consumer GPUs is borderline.** The model card claims RTF ~0.3 on an RTX 4090; on a 3060 we measure RTF ~1.5–2 even with `torch.compile` + Triton. That means audio for long sentences may have small gaps as the buffer drains while the next chunk is still generating. Lowering `voxcpm_inference_timesteps` helps. For latency-critical conversations, swap to ElevenLabs or Azure via the dropdown.
- **Languages**: the model card lists 30 languages including Arabic and Hindi but does *not* explicitly list Urdu. It auto-detects language from text. Quality on Urdu is unverified — try and see.
- **First run** downloads ~5 GB of weights and JIT-compiles CUDA kernels (~10 s on first turn after model load). After that, the model stays in memory across pipeline rebuilds.

## Project layout

```
.
├── ui.py                # CustomTkinter desktop UI (entrypoint)
├── bot.py               # CLI entrypoint — same pipeline, no UI
├── controller.py        # asyncio worker thread that owns the live pipeline
├── pipeline.py          # transport-agnostic Pipecat pipeline
├── config.py            # Config dataclass, load_config(), save_env()
├── services/
│   ├── stt.py           # STT provider factory
│   ├── llm.py           # LLM provider factory
│   ├── tts.py           # TTS provider factory
│   └── voxcpm_tts.py    # custom Pipecat TTS wrapping openbmb/VoxCPM2
└── transports/
    ├── local.py         # local mic + speakers (PyAudio)
    └── webrtc.py        # SmallWebRTCTransport stub for the future web UI
```

## Roadmap

- [x] Desktop UI with live provider/model/key editing
- [x] Local TTS via VoxCPM2 with `torch.compile` + Triton acceleration
- [ ] Conversation logging / transcript export
- [ ] Per-turn metrics shown in the UI (STT/LLM/TTS latency)
- [ ] STT model caching across pipeline rebuilds (Whisper currently reloads on each Apply)
