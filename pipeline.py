"""Builds the Pipecat pipeline. Transport-agnostic — the transport is injected."""

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from config import Config
from services.stt import create_stt
from services.llm import create_llm
from services.tts import create_tts


def build_pipeline(transport, cfg: Config, messages: list | None = None):
    """
    Build a fresh PipelineTask from the given config and seed message history.

    Returns (task, context). The caller can read `context.messages` after a
    conversation to extract turn history for a subsequent rebuild.
    """
    stt = create_stt(cfg)
    llm = create_llm(cfg)
    tts = create_tts(cfg)

    # Always pin the first message to the current system prompt so prompt edits
    # take effect after a rebuild. Preserve any user/assistant turns that follow.
    history = []
    if messages:
        history = [m for m in messages if m.get("role") != "system"]
    seeded = [{"role": "system", "content": cfg.system_prompt}] + history

    context = OpenAILLMContext(seeded)
    aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregator.user(),
            llm,
            tts,
            transport.output(),
            aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )
    return task, context
