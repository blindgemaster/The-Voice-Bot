"""
Controller — owns the asyncio worker thread that drives the Pipecat pipeline.

The Tk main thread interacts via three thread-safe entry points: start(),
stop(), apply_config(new_cfg). Events (status changes, transcripts, errors)
flow back to the UI through a queue.Queue that the UI polls with root.after().
"""

import asyncio
import queue
import threading
import traceback

from pipecat.pipeline.runner import PipelineRunner

from config import Config
from pipeline import build_pipeline
from transports.local import get_transport


class Controller:
    def __init__(self, cfg: Config, event_q: queue.Queue):
        self._cfg = cfg
        self._event_q = event_q

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()

        self._task = None
        self._runner = None
        self._transport = None
        self._context = None
        self._running_task: asyncio.Task | None = None

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._loop_ready.wait()

    # ---- worker thread bootstrap ----

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _emit(self, kind: str, *payload):
        self._event_q.put((kind, *payload))

    # ---- public API (called from Tk thread) ----

    def start(self):
        asyncio.run_coroutine_threadsafe(self._start(), self._loop)

    def stop(self):
        asyncio.run_coroutine_threadsafe(self._stop(), self._loop)

    def apply_config(self, cfg: Config):
        asyncio.run_coroutine_threadsafe(self._apply(cfg), self._loop)

    def shutdown(self):
        try:
            fut = asyncio.run_coroutine_threadsafe(self._stop(), self._loop)
            fut.result(timeout=3)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)

    # ---- coroutines (run on worker loop) ----

    async def _start(self):
        if self._task is not None:
            return
        await self._build_and_run(messages=None)

    async def _stop(self):
        if self._task is None:
            return
        try:
            await self._task.cancel()
        except Exception:
            pass
        if self._running_task is not None:
            try:
                await asyncio.wait_for(self._running_task, timeout=3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                pass
        self._task = None
        self._runner = None
        self._transport = None
        self._context = None
        self._running_task = None
        self._emit("status", "idle")

    async def _apply(self, cfg: Config):
        self._cfg = cfg
        if self._task is None:
            # Not running — config is now staged for the next start.
            return
        self._emit("status", "rebuilding")
        prior = []
        if self._context is not None:
            try:
                prior = list(self._context.messages)
            except Exception:
                prior = []
        await self._stop()
        await self._build_and_run(messages=prior)

    async def _build_and_run(self, messages):
        try:
            self._emit("status", "starting")
            self._transport = get_transport()
            self._task, self._context = build_pipeline(self._transport, self._cfg, messages)
            # PipelineRunner's default SIGINT handler can't be installed off the
            # main thread (Windows asyncio + signal module both refuse). Skip it.
            self._runner = PipelineRunner(handle_sigint=False)
            self._running_task = asyncio.create_task(self._run_safely())
            self._emit("status", "running")
        except Exception:
            self._emit("error", traceback.format_exc())
            self._emit("status", "idle")
            self._task = None
            self._runner = None
            self._transport = None
            self._context = None
            self._running_task = None

    async def _run_safely(self):
        try:
            await self._runner.run(self._task)
        except asyncio.CancelledError:
            pass
        except Exception:
            self._emit("error", traceback.format_exc())

