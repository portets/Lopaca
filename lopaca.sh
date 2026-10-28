#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Run install.sh for a menu shortcut, or use this script in a terminal to debug.
exec python3 main.py "$@"
