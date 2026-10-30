"""Media engines. Both speak WHIP (cameras publish) and WHEP (viewers play).

- ``pyrtc``: built-in aiortc engine. No external binary, fine for a few cameras and for tests.
- ``mediamtx``: production engine. The hub downloads the official MediaMTX release for the
  platform on first run, verifies its checksum, and runs it as a child process.
"""
from .base import MediaEngine, StreamInfo

__all__ = ["MediaEngine", "StreamInfo"]
