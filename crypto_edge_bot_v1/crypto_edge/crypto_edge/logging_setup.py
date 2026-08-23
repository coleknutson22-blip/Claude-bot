"""Structured logging with separate channels so a post-mortem can isolate
market-data failures from strategy decisions from execution."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path

CHANNELS = ("app", "data", "strategy", "trades", "performance", "telegram")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "channel": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-11s %(message)s",
                         "%H:%M:%S")


def setup_logging(log_dir: str = "logs", level: str = "INFO",
                  console: bool = True) -> None:
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("ce")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(HumanFormatter())
        root.addHandler(ch)

    for channel in CHANNELS:
        lg = logging.getLogger(f"ce.{channel}")
        lg.setLevel(logging.DEBUG)
        fh = logging.handlers.RotatingFileHandler(
            d / f"{channel}.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        lg.addHandler(fh)


def get_logger(channel: str = "app") -> logging.Logger:
    return logging.getLogger(f"ce.{channel}")


def log_event(channel: str, level: str, msg: str, **fields) -> None:
    lg = get_logger(channel)
    lg.log(getattr(logging, level.upper(), logging.INFO), msg,
           extra={"extra_fields": fields})
