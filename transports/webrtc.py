"""WebRTC transport — swap in when the web UI is ready.

Requires: pip install "pipecat-ai[webrtc]"

This will use Pipecat's SmallWebRTCTransport for a fully self-hosted
browser-to-bot connection (no Daily.co dependency).

Usage (in bot.py):
    from transports.webrtc import get_transport, get_app
    transport = get_transport(...)
    app = get_app()  # FastAPI/Starlette app to serve signaling
"""

# Stub — uncomment and flesh out when building the web UI.
#
# from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
# from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
#
#
# def get_transport(webrtc_connection):
#     return SmallWebRTCTransport(
#         webrtc_connection=webrtc_connection,
#         params=TransportParams(
#             audio_in_enabled=True,
#             audio_out_enabled=True,
#             vad_analyzer=SileroVADAnalyzer(),
#         ),
#     )


def get_transport():
    raise NotImplementedError(
        "WebRTC transport is not yet implemented. "
        "Use transports.local for now."
    )
