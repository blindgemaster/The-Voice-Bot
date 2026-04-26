"""Local audio transport — uses system mic and speakers via PyAudio."""

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)


def get_transport():
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )
