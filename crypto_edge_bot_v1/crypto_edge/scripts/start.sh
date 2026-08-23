#!/usr/bin/env bash
# Start the paper trader in the background, with a PID file and a log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data
if [ -f logs/bot.pid ] && kill -0 "$(cat logs/bot.pid)" 2>/dev/null; then
  echo "Already running (pid $(cat logs/bot.pid)). Use scripts/stop.sh first."
  exit 1
fi
echo "Running self-check before start..."
python3 -m crypto_edge.cli selfcheck || { echo "Self-check FAILED. Not starting."; exit 1; }
nohup python3 -m crypto_edge.cli start >> logs/bot.out 2>&1 &
echo $! > logs/bot.pid
echo "Started (pid $!). Follow with: tail -f logs/bot.out"
