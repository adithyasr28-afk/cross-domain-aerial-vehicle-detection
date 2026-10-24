#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 app.py --server-name "${DEMO_HOST:-127.0.0.1}" --server-port "${DEMO_PORT:-7860}"
