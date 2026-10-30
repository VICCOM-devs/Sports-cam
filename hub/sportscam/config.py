"""Runtime configuration. Everything lives under one data directory so a hub is portable."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    env = os.environ.get("SPORTSCAM_DATA")
    if env:
        return Path(env)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "VICCOM" / "SportsCam"
    return Path.home() / ".sportscam"


def lan_ip() -> str:
    """Best-effort LAN address, used in the QR code. Never actually sends a packet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@dataclass
class Config:
    data_dir: Path = field(default_factory=_default_data_dir)
    http_port: int = int(os.environ.get("SPORTSCAM_HTTP_PORT", "8080"))
    https_port: int = int(os.environ.get("SPORTSCAM_HTTPS_PORT", "8443"))
    public_host: str = os.environ.get("SPORTSCAM_HOST", "") or lan_ip()
    # media engine: "auto" picks mediamtx when its binary is present, else the built-in aiortc engine
    media_engine: str = os.environ.get("SPORTSCAM_MEDIA", "auto")
    mediamtx_api: str = "http://127.0.0.1:9997"
    mediamtx_webrtc: str = "http://127.0.0.1:8889"
    update_url: str = os.environ.get(
        "SPORTSCAM_UPDATE_URL", "https://viccom.se/updates/sports-cam.json"
    )
    record: bool = os.environ.get("SPORTSCAM_RECORD", "1") != "0"
    vision: bool = os.environ.get("SPORTSCAM_VISION", "1") != "0"
    vision_model: str = os.environ.get("SPORTSCAM_MODEL", "nano")      # nano | s
    vision_fps: float = float(os.environ.get("SPORTSCAM_VISION_FPS", "5"))
    models_dir_override: str = os.environ.get("SPORTSCAM_MODELS", "")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sportscam.sqlite"

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def certs_dir(self) -> Path:
        return self.data_dir / "certs"

    @property
    def bin_dir(self) -> Path:
        return self.data_dir / "bin"

    @property
    def models_dir(self) -> Path:
        return Path(self.models_dir_override) if self.models_dir_override else self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.recordings_dir, self.certs_dir, self.bin_dir):
            d.mkdir(parents=True, exist_ok=True)
