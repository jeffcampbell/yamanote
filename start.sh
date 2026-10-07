#!/bin/bash
# Yamanote launcher — restarts automatically on exit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Yamanote starting..."

while true; do
    [ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a
    python3 -m yamanote "$@"
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Yamanote exited (rc=$EXIT_CODE), restarting in 3 seconds..."
    sleep 3
done
