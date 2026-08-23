#!/usr/bin/env bash
# Graceful stop: the engine finishes its current cycle, then persists and exits.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f logs/bot.pid ]; then echo "No pid file; nothing to stop."; exit 0; fi
PID=$(cat logs/bot.pid)
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID not running; clearing stale pid file."; rm -f logs/bot.pid; exit 0
fi
echo "Sending SIGTERM to $PID (finishes current cycle, then exits)..."
kill -TERM "$PID"
for _ in $(seq 1 60); do
  kill -0 "$PID" 2>/dev/null || { echo "Stopped cleanly."; rm -f logs/bot.pid; exit 0; }
  sleep 1
done
echo "Still running after 60s. Inspect manually before using kill -9;"
echo "state is committed per-cycle, so a hard kill is safe but may lose the"
echo "cycle in flight."
exit 1
