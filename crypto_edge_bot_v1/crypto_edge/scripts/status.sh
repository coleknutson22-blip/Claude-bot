#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f logs/bot.pid ] && kill -0 "$(cat logs/bot.pid)" 2>/dev/null; then
  echo "Process: RUNNING (pid $(cat logs/bot.pid))"
else
  echo "Process: NOT RUNNING"
fi
echo
python3 -m crypto_edge.cli status
