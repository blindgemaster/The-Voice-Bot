"""Entrypoint — runs the Urdu voice bot with local audio."""

import glob
import os
import sys

# Make NVIDIA CUDA DLLs (cuBLAS, cuDNN) discoverable on Windows
nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
nvidia_bins = glob.glob(os.path.join(nvidia_base, "*", "bin"))
if nvidia_bins:
    os.environ["PATH"] = os.pathsep.join(nvidia_bins) + os.pathsep + os.environ.get("PATH", "")
    for bin_dir in nvidia_bins:
        os.add_dll_directory(bin_dir)

import asyncio

from pipecat.pipeline.runner import PipelineRunner

from transports.local import get_transport
from pipeline import build_pipeline


async def main():
    transport = get_transport()
    task = build_pipeline(transport)

    runner = PipelineRunner()

    print("Listening... speak in Urdu or English")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
