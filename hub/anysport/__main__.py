"""Run the hub: ``python -m anysport``.

Serves HTTP (viewers, lobby) and HTTPS (camera phones, which need a secure origin) at once.
"""
from __future__ import annotations

import asyncio
import logging

import uvicorn

from .certs import ensure_cert
from .config import Config
from .main import create_app


async def serve(cfg: Config) -> None:
    app = create_app(cfg)
    cert, key = ensure_cert(cfg.certs_dir, cfg.public_host)
    http = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=cfg.http_port, log_level="info", lifespan="on"))
    https = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=cfg.https_port, log_level="warning",
                                          ssl_certfile=str(cert), ssl_keyfile=str(key), lifespan="off"))
    await asyncio.gather(http.serve(), https.serve())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(serve(Config()))


if __name__ == "__main__":
    main()
