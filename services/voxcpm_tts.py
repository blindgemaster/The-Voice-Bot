"""
Pipecat TTS service for openbmb/VoxCPM2 — a local 2B-parameter diffusion TTS.

Loads the model once per process (cached at module level so pipeline rebuilds
don't re-pay the ~20s/8GB-VRAM load cost). Streams generated audio chunks if
the model supports it, otherwise falls back to whole-utterance generation.

Output is float32 mono at 48kHz; we resample to the configured rate and
convert to int16 PCM for Pipecat's `TTSAudioRawFrame`.

Threading: torch.compile + CUDA graphs bind to the thread that performs the
compile via thread-local state. To keep that state consistent we run model
load, warmup, and every inference call on a single dedicated worker thread
(`_EXECUTOR`, `max_workers=1`). asyncio.to_thread (which uses the default
threadpool) violates that invariant and trips an internal torch assertion.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import AsyncGenerator

import numpy as np
import torch
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

# Each new input shape triggers a fresh CUDA graph recording, which is
# expensive. With short, varied LLM replies we hit ~10 distinct shapes per
# minute and pay the recompile tax repeatedly. Skipping dynamic graphs trades
# a small per-call speedup for predictable steady-state latency.
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True


_MODEL = None  # process-wide cache; loaded on first use
_LOAD_LOCK = threading.Lock()
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="voxcpm"
)


def _load_model_sync():
    """Lazy-load VoxCPM2 once. Always invoked on the dedicated executor thread."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        from voxcpm import VoxCPM
        logger.info("VoxCPM: loading openbmb/VoxCPM2 (first call, this can take a while)...")
        _MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        logger.info("VoxCPM: model loaded")
        return _MODEL


def _resample_and_quantize(chunk: np.ndarray, in_sr: int, out_sr: int) -> bytes:
    """Float32 mono [-1, 1] at in_sr -> int16 PCM bytes at out_sr."""
    if chunk.ndim > 1:
        chunk = chunk.squeeze()
    if in_sr != out_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(in_sr, out_sr)
        chunk = resample_poly(chunk, out_sr // g, in_sr // g)
    chunk = np.clip(chunk, -1.0, 1.0)
    return (chunk * 32767.0).astype(np.int16).tobytes()


class VoxCPMTTSService(TTSService):
    """Local TTS via the `voxcpm` package (openbmb/VoxCPM2)."""

    def __init__(
        self,
        *,
        reference_wav: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )
        self._reference_wav = reference_wav.strip() or None
        self._cfg_value = cfg_value
        self._inference_timesteps = inference_timesteps
        # Preload synchronously on the dedicated executor thread so the first
        # run_tts call doesn't race against Pipecat's TTS context timeout while
        # the 8GB model is initializing. Subsequent rebuilds reuse the cache.
        _EXECUTOR.submit(_load_model_sync).result()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = text.strip()
        if not text:
            return

        logger.debug(f"{self}: VoxCPM TTS [{text}]")
        loop = asyncio.get_running_loop()

        try:
            model = await loop.run_in_executor(_EXECUTOR, _load_model_sync)
        except Exception as e:
            logger.exception("VoxCPM model load failed")
            yield ErrorFrame(error=f"VoxCPM model load failed: {e}")
            return

        in_sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", 48000))
        out_sr = self.sample_rate

        gen_kwargs = {
            "text": text,
            "cfg_value": self._cfg_value,
            "inference_timesteps": self._inference_timesteps,
        }
        if self._reference_wav:
            gen_kwargs["reference_wav_path"] = self._reference_wav

        await self.start_tts_usage_metrics(text)

        try:
            if hasattr(model, "generate_streaming"):
                async for frame in self._stream(loop, model, gen_kwargs, in_sr, out_sr, context_id):
                    yield frame
            else:
                async for frame in self._oneshot(loop, model, gen_kwargs, in_sr, out_sr, context_id):
                    yield frame
        except Exception as e:
            logger.exception("VoxCPM TTS failure")
            yield ErrorFrame(error=f"VoxCPM TTS error: {e}")

    async def _stream(self, loop, model, gen_kwargs, in_sr, out_sr, context_id):
        # Build the generator on the executor thread too — VoxCPM may touch
        # CUDA tensors during generator construction.
        gen = await loop.run_in_executor(
            _EXECUTOR, lambda: model.generate_streaming(**gen_kwargs)
        )
        sentinel = object()

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        first = True
        while True:
            chunk = await loop.run_in_executor(_EXECUTOR, _next)
            if chunk is sentinel:
                break
            if chunk is None or len(chunk) == 0:
                continue
            audio_bytes = _resample_and_quantize(np.asarray(chunk, dtype=np.float32), in_sr, out_sr)
            if first:
                await self.stop_ttfb_metrics()
                first = False
            yield TTSAudioRawFrame(audio_bytes, out_sr, 1, context_id=context_id)

    async def _oneshot(self, loop, model, gen_kwargs, in_sr, out_sr, context_id):
        wav = await loop.run_in_executor(
            _EXECUTOR, lambda: model.generate(**gen_kwargs)
        )
        await self.stop_ttfb_metrics()
        audio = _resample_and_quantize(np.asarray(wav, dtype=np.float32), in_sr, out_sr)
        bytes_per_chunk = max(2, (out_sr // 20) * 2)  # ~50ms of 16-bit mono
        for i in range(0, len(audio), bytes_per_chunk):
            yield TTSAudioRawFrame(audio[i:i + bytes_per_chunk], out_sr, 1, context_id=context_id)
