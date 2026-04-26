"""CustomTkinter desktop UI for The Voice Bot.

Tkinter event loop runs on the main thread. The Pipecat pipeline runs on a
worker asyncio loop owned by `controller.Controller`. Communication is via
two queues: commands flow UI → controller via direct method calls (which
schedule coroutines on the worker loop), events flow controller → UI through
a thread-safe queue polled with `root.after()`.
"""

import glob
import os
import queue
import sys

# Make NVIDIA CUDA DLLs (cuBLAS, cuDNN) discoverable on Windows.
_nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
_nvidia_bins = glob.glob(os.path.join(_nvidia_base, "*", "bin"))
if _nvidia_bins:
    os.environ["PATH"] = os.pathsep.join(_nvidia_bins) + os.pathsep + os.environ.get("PATH", "")
    for _b in _nvidia_bins:
        os.add_dll_directory(_b)

import customtkinter as ctk

from config import (
    Config,
    LLMProvider,
    STTProvider,
    TTSProvider,
    load_config,
    save_env,
    save_state,
)
from controller import Controller


PROVIDER_LABELS = {
    STTProvider.WHISPER_LOCAL: "Whisper (local GPU)",
    STTProvider.WHISPER_CLOUD: "OpenAI Whisper (cloud)",
    STTProvider.AZURE: "Azure",
    STTProvider.DEEPGRAM: "Deepgram",
    LLMProvider.OLLAMA: "Ollama",
    LLMProvider.LMSTUDIO: "LM Studio",
    LLMProvider.OPENAI: "OpenAI",
    LLMProvider.ANTHROPIC: "Anthropic",
    TTSProvider.AZURE: "Azure",
    TTSProvider.ELEVENLABS: "ElevenLabs",
    TTSProvider.OPENAI: "OpenAI",
    TTSProvider.CARTESIA: "Cartesia",
    TTSProvider.VOXCPM: "VoxCPM2 (local GPU)",
}

STT_LABELS = [PROVIDER_LABELS[p] for p in STTProvider]
LLM_LABELS = [PROVIDER_LABELS[p] for p in LLMProvider]
# TTS labels need to be unique strings; Azure/OpenAI appear in multiple enums,
# so we disambiguate at lookup time using per-stage maps.
TTS_LABELS = [PROVIDER_LABELS[p] for p in TTSProvider]

LABEL_TO_STT = {PROVIDER_LABELS[p]: p for p in STTProvider}
LABEL_TO_LLM = {PROVIDER_LABELS[p]: p for p in LLMProvider}
LABEL_TO_TTS = {PROVIDER_LABELS[p]: p for p in TTSProvider}


STATUS_COLORS = {
    "idle":       "#6b7280",
    "starting":   "#f59e0b",
    "running":    "#10b981",
    "listening":  "#10b981",
    "thinking":   "#f59e0b",
    "speaking":   "#3b82f6",
    "rebuilding": "#a855f7",
    "error":      "#ef4444",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large_v3_turbo"]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("The Voice Bot")
        self.geometry("780x920")
        self.minsize(720, 700)

        self.cfg: Config = load_config()
        self.event_q: queue.Queue = queue.Queue()
        self.controller = Controller(self.cfg, self.event_q)

        self._build()
        self._sync_from_cfg()
        self.after(50, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- layout ----

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header, text="The Voice Bot", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=12, pady=10, sticky="w")
        self.status_pill = ctk.CTkLabel(
            header,
            text="● idle",
            font=ctk.CTkFont(size=13),
            text_color=STATUS_COLORS["idle"],
        )
        self.status_pill.grid(row=0, column=1, padx=12, pady=10, sticky="e")

        self._build_providers().grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        self._build_keys().grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        self._build_prompt().grid(row=3, column=0, padx=12, pady=6, sticky="ew")
        self._build_transcript().grid(row=4, column=0, padx=12, pady=6, sticky="nsew")
        self._build_controls().grid(row=5, column=0, padx=12, pady=(6, 12), sticky="ew")

    def _build_providers(self):
        f = ctk.CTkFrame(self)
        f.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(
            f, text="Providers", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=12, pady=(10, 6), sticky="w")

        ctk.CTkLabel(f, text="STT").grid(row=1, column=0, padx=(12, 6), pady=4, sticky="w")
        self.stt_menu = ctk.CTkOptionMenu(f, values=STT_LABELS, command=self._on_stt_changed)
        self.stt_menu.grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        ctk.CTkLabel(f, text="Model").grid(row=1, column=2, padx=(12, 6), pady=4, sticky="w")
        self.stt_model_entry = ctk.CTkEntry(f)
        self.stt_model_entry.grid(row=1, column=3, padx=(6, 12), pady=4, sticky="ew")

        ctk.CTkLabel(f, text="LLM").grid(row=2, column=0, padx=(12, 6), pady=4, sticky="w")
        self.llm_menu = ctk.CTkOptionMenu(f, values=LLM_LABELS, command=self._on_llm_changed)
        self.llm_menu.grid(row=2, column=1, padx=6, pady=4, sticky="ew")
        ctk.CTkLabel(f, text="Model").grid(row=2, column=2, padx=(12, 6), pady=4, sticky="w")
        self.llm_model_entry = ctk.CTkEntry(f)
        self.llm_model_entry.grid(row=2, column=3, padx=(6, 12), pady=4, sticky="ew")

        ctk.CTkLabel(f, text="TTS").grid(row=3, column=0, padx=(12, 6), pady=4, sticky="w")
        self.tts_menu = ctk.CTkOptionMenu(f, values=TTS_LABELS, command=self._on_tts_changed)
        self.tts_menu.grid(row=3, column=1, padx=6, pady=4, sticky="ew")
        ctk.CTkLabel(f, text="Voice/Ref").grid(row=3, column=2, padx=(12, 6), pady=4, sticky="w")
        self.tts_voice_entry = ctk.CTkEntry(f)
        self.tts_voice_entry.grid(row=3, column=3, padx=(6, 12), pady=4, sticky="ew")

        ctk.CTkLabel(f, text="Lang").grid(row=4, column=0, padx=(12, 6), pady=4, sticky="w")
        self.lang_entry = ctk.CTkEntry(f, width=120)
        self.lang_entry.grid(row=4, column=1, padx=6, pady=4, sticky="w")

        ctk.CTkButton(f, text="Apply", width=110, command=self._on_apply).grid(
            row=4, column=3, padx=(6, 12), pady=(4, 12), sticky="e"
        )
        return f

    def _build_keys(self):
        f = ctk.CTkFrame(self)
        f.grid_columnconfigure(1, weight=1)
        f.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(
            f, text="API keys & endpoints", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=12, pady=(10, 6), sticky="w")

        rows = [
            ("openai_api_key",      "OpenAI API key",     True),
            ("anthropic_api_key",   "Anthropic API key",  True),
            ("azure_speech_key",    "Azure Speech key",   True),
            ("azure_speech_region", "Azure region",       False),
            ("deepgram_api_key",    "Deepgram API key",   True),
            ("elevenlabs_api_key",  "ElevenLabs API key", True),
            ("elevenlabs_voice_id", "ElevenLabs voice ID", False),
            ("cartesia_api_key",    "Cartesia API key",   True),
            ("cartesia_voice_id",   "Cartesia voice ID",  False),
            ("ollama_base_url",     "Ollama base URL",    False),
            ("lmstudio_base_url",   "LM Studio base URL", False),
        ]
        self._key_entries: dict[str, ctk.CTkEntry] = {}
        # Two-column layout for compactness.
        for i, (attr, label, mask) in enumerate(rows):
            col = (i % 2) * 2
            row = 1 + i // 2
            ctk.CTkLabel(f, text=label).grid(row=row, column=col, padx=(12, 6), pady=2, sticky="w")
            entry = ctk.CTkEntry(f, show="•" if mask else "")
            entry.grid(row=row, column=col + 1, padx=(6, 12), pady=2, sticky="ew")
            self._key_entries[attr] = entry

        last = 1 + (len(rows) + 1) // 2
        ctk.CTkButton(f, text="Save to .env", width=140, command=self._on_save_keys).grid(
            row=last, column=3, padx=(6, 12), pady=(8, 12), sticky="e"
        )
        return f

    def _build_prompt(self):
        f = ctk.CTkFrame(self)
        f.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            f, text="System prompt", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, padx=12, pady=(10, 6), sticky="w")
        self.prompt_box = ctk.CTkTextbox(f, height=80, wrap="word")
        self.prompt_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        return f

    def _build_transcript(self):
        f = ctk.CTkFrame(self)
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            f, text="Transcript", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, padx=12, pady=(10, 6), sticky="w")
        self.transcript_box = ctk.CTkTextbox(f, wrap="word")
        self.transcript_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.transcript_box.configure(state="disabled")
        return f

    def _build_controls(self):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            f, text="Start", height=40, command=self._on_start
        ).grid(row=0, column=0, padx=6, sticky="ew")
        ctk.CTkButton(
            f, text="Stop", height=40, fg_color="#7c2d2d", hover_color="#992c2c",
            command=self._on_stop,
        ).grid(row=0, column=1, padx=6, sticky="ew")
        return f

    # ---- UI <-> cfg sync ----

    def _stt_model_for(self, p: STTProvider) -> str:
        return self.cfg.local_whisper_model if p == STTProvider.WHISPER_LOCAL else ""

    def _llm_model_for(self, p: LLMProvider) -> str:
        return {
            LLMProvider.OLLAMA:    self.cfg.ollama_model,
            LLMProvider.LMSTUDIO:  self.cfg.lmstudio_model,
            LLMProvider.OPENAI:    self.cfg.openai_llm_model,
            LLMProvider.ANTHROPIC: self.cfg.anthropic_llm_model,
        }[p]

    def _tts_voice_for(self, p: TTSProvider) -> str:
        return {
            TTSProvider.AZURE:      self.cfg.azure_tts_voice,
            TTSProvider.ELEVENLABS: self.cfg.elevenlabs_voice_id,
            TTSProvider.OPENAI:     self.cfg.openai_tts_voice,
            TTSProvider.CARTESIA:   self.cfg.cartesia_voice_id,
            TTSProvider.VOXCPM:     self.cfg.voxcpm_reference_wav,
        }[p]

    def _set_entry(self, entry: ctk.CTkEntry, value: str):
        entry.delete(0, "end")
        entry.insert(0, value or "")

    def _sync_from_cfg(self):
        self.stt_menu.set(PROVIDER_LABELS[self.cfg.stt_provider])
        self.llm_menu.set(PROVIDER_LABELS[self.cfg.llm_provider])
        self.tts_menu.set(PROVIDER_LABELS[self.cfg.tts_provider])

        self._set_entry(self.stt_model_entry, self._stt_model_for(self.cfg.stt_provider))
        self._set_entry(self.llm_model_entry, self._llm_model_for(self.cfg.llm_provider))
        self._set_entry(self.tts_voice_entry, self._tts_voice_for(self.cfg.tts_provider))
        self._set_entry(self.lang_entry, self.cfg.stt_language)

        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", self.cfg.system_prompt)

        for attr, entry in self._key_entries.items():
            self._set_entry(entry, getattr(self.cfg, attr))

    def _on_stt_changed(self, label: str):
        p = LABEL_TO_STT[label]
        self._set_entry(self.stt_model_entry, self._stt_model_for(p))

    def _on_llm_changed(self, label: str):
        p = LABEL_TO_LLM[label]
        self._set_entry(self.llm_model_entry, self._llm_model_for(p))

    def _on_tts_changed(self, label: str):
        p = LABEL_TO_TTS[label]
        self._set_entry(self.tts_voice_entry, self._tts_voice_for(p))

    def _read_cfg_from_ui(self) -> Config:
        cfg = Config(**self.cfg.__dict__)

        cfg.stt_provider = LABEL_TO_STT[self.stt_menu.get()]
        cfg.llm_provider = LABEL_TO_LLM[self.llm_menu.get()]
        cfg.tts_provider = LABEL_TO_TTS[self.tts_menu.get()]
        cfg.stt_language = self.lang_entry.get().strip() or "ur"

        if cfg.stt_provider == STTProvider.WHISPER_LOCAL:
            v = self.stt_model_entry.get().strip()
            if v in WHISPER_MODELS:
                cfg.local_whisper_model = v
            elif v:
                cfg.local_whisper_model = v  # let user override at their own risk

        llm_model = self.llm_model_entry.get().strip()
        if llm_model:
            if cfg.llm_provider == LLMProvider.OLLAMA:    cfg.ollama_model = llm_model
            if cfg.llm_provider == LLMProvider.LMSTUDIO:  cfg.lmstudio_model = llm_model
            if cfg.llm_provider == LLMProvider.OPENAI:    cfg.openai_llm_model = llm_model
            if cfg.llm_provider == LLMProvider.ANTHROPIC: cfg.anthropic_llm_model = llm_model

        tts_voice = self.tts_voice_entry.get().strip()
        # VoxCPM treats "voice" as an optional reference wav path (empty allowed).
        if cfg.tts_provider == TTSProvider.VOXCPM:
            cfg.voxcpm_reference_wav = tts_voice
        elif tts_voice:
            if cfg.tts_provider == TTSProvider.AZURE:      cfg.azure_tts_voice = tts_voice
            if cfg.tts_provider == TTSProvider.ELEVENLABS: cfg.elevenlabs_voice_id = tts_voice
            if cfg.tts_provider == TTSProvider.OPENAI:     cfg.openai_tts_voice = tts_voice
            if cfg.tts_provider == TTSProvider.CARTESIA:   cfg.cartesia_voice_id = tts_voice

        prompt = self.prompt_box.get("1.0", "end").strip()
        if prompt:
            cfg.system_prompt = prompt

        for attr, entry in self._key_entries.items():
            setattr(cfg, attr, entry.get())

        return cfg

    # ---- button handlers ----

    def _on_apply(self):
        self.cfg = self._read_cfg_from_ui()
        # Push edited keys into os.environ so newly-configured providers can pick
        # them up on rebuild without requiring a separate Save first.
        for attr in ("openai_api_key", "anthropic_api_key", "azure_speech_key",
                     "azure_speech_region", "deepgram_api_key",
                     "elevenlabs_api_key", "elevenlabs_voice_id",
                     "cartesia_api_key", "cartesia_voice_id",
                     "ollama_base_url", "lmstudio_base_url"):
            os.environ[attr.upper()] = getattr(self.cfg, attr) or ""
        # Persist UI-managed settings so the next launch loads what you applied,
        # not the dataclass defaults.
        try:
            save_state(self.cfg)
        except Exception as e:
            self._flash_status("error", f"state save failed: {e}")
        self.controller.apply_config(self.cfg)

    def _on_save_keys(self):
        self.cfg = self._read_cfg_from_ui()
        try:
            save_env(self.cfg)
            self._flash_status("idle", "saved to .env")
        except Exception as e:
            self._flash_status("error", f"save failed: {e}")

    def _on_start(self):
        self.cfg = self._read_cfg_from_ui()
        self.controller.apply_config(self.cfg)  # stages cfg if not running
        self.controller.start()

    def _on_stop(self):
        self.controller.stop()

    def _on_close(self):
        try:
            self.controller.shutdown()
        finally:
            self.destroy()

    # ---- event polling (controller -> UI) ----

    def _poll_events(self):
        try:
            while True:
                ev = self.event_q.get_nowait()
                kind = ev[0]
                if kind == "status":
                    self._set_status(ev[1])
                elif kind == "transcript":
                    self._append_transcript(ev[1], ev[2])
                elif kind == "error":
                    self._set_status("error")
                    self._append_transcript("error", ev[1])
        except queue.Empty:
            pass
        self.after(50, self._poll_events)

    def _set_status(self, status: str):
        color = STATUS_COLORS.get(status, STATUS_COLORS["idle"])
        self.status_pill.configure(text=f"● {status}", text_color=color)

    def _flash_status(self, status: str, text: str):
        color = STATUS_COLORS.get(status, STATUS_COLORS["idle"])
        self.status_pill.configure(text=f"● {text}", text_color=color)
        self.after(1500, lambda: self._set_status(status))

    def _append_transcript(self, role: str, text: str):
        prefix = {"user": "You: ", "assistant": "Bot: ", "error": "Error: "}.get(role, "")
        self.transcript_box.configure(state="normal")
        self.transcript_box.insert("end", f"{prefix}{text}\n\n")
        self.transcript_box.see("end")
        self.transcript_box.configure(state="disabled")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
