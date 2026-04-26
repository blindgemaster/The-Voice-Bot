"""Builds the Pipecat pipeline. Transport-agnostic — the transport is injected."""

import config
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from services.stt import create_stt
from services.llm import create_llm
from services.tts import create_tts


def build_pipeline(transport) -> PipelineTask:
    stt = create_stt()
    llm = create_llm()
    tts = create_tts()

    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    return PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )
