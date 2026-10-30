#!/usr/bin/env bash
# VICCOM Sports Cam hub – macOS/Linux launcher
cd "$(dirname "$0")/hub"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
exec .venv/bin/python -m sportscam
