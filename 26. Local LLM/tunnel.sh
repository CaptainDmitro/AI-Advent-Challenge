#!/usr/bin/env bash
# Lets the lesson deployed on the VDS use Ollama on this Mac. A reverse SSH
# tunnel makes this Mac's 127.0.0.1:11434 show up as the VDS's 127.0.0.1:11434,
# which is the app's default OLLAMA_HOST. Nothing is exposed publicly: on the
# VDS the port is bound to 127.0.0.1 only, and the Mac only connects out.
#
# Usage: ./tunnel.sh user@vds-host        (Ctrl+C to stop)
set -euo pipefail

TARGET="${1:?usage: ./tunnel.sh user@vds-host}"
PORT="${OLLAMA_PORT:-11434}"

if ! curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null; then
  echo "Ollama isn't answering on 127.0.0.1:$PORT. Start it: brew services start ollama" >&2
  exit 1
fi

# Reconnects after sleep, Wi-Fi changes or a VDS reboot.
while true; do
  echo "$(date '+%H:%M:%S') connecting: $TARGET 127.0.0.1:$PORT -> this Mac 127.0.0.1:$PORT"
  ssh -N \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:$PORT:127.0.0.1:$PORT" \
    "$TARGET" || true
  echo "$(date '+%H:%M:%S') tunnel dropped, reconnecting in 5 s (Ctrl+C to stop)"
  sleep 5
done
