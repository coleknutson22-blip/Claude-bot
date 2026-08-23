"""UTC-only time handling. Every timestamp in this system is milliseconds since
the Unix epoch, UTC. No naive datetimes, no local timezones, ever."""
from __future__ import annotations

from datetime import datetime, timezone

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS

TIMEFRAME_MS = {
    "1m": MINUTE_MS, "3m": 3 * MINUTE_MS, "5m": 5 * MINUTE_MS,
    "15m": 15 * MINUTE_MS, "30m": 30 * MINUTE_MS,
    "1h": HOUR_MS, "2h": 2 * HOUR_MS, "4h": 4 * HOUR_MS,
    "6h": 6 * HOUR_MS, "12h": 12 * HOUR_MS, "1d": DAY_MS,
}


def tf_ms(timeframe: str) -> int:
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise ValueError(f"unsupported timeframe: {timeframe}") from None


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def iso(ms: int) -> str:
    return to_dt(ms).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date(ms: int) -> str:
    return to_dt(ms).strftime("%Y-%m-%d")


def floor_to_tf(ms: int, timeframe: str) -> int:
    step = tf_ms(timeframe)
    return (ms // step) * step


def candle_close_ms(open_ms: int, timeframe: str) -> int:
    """A candle stamped with open time `open_ms` is only complete at this instant."""
    return open_ms + tf_ms(timeframe)


def is_closed(open_ms: int, timeframe: str, now: int, buffer_ms: int = 0) -> bool:
    """True only if the candle has genuinely finished.

    `buffer_ms` guards against exchange clock skew and late-arriving ticks: we
    refuse to call a candle closed until a little after its nominal close.
    """
    return now >= candle_close_ms(open_ms, timeframe) + buffer_ms


def last_closed_open_ms(timeframe: str, now: int, buffer_ms: int = 0) -> int:
    """Open timestamp of the most recent candle that is definitely complete."""
    step = tf_ms(timeframe)
    return floor_to_tf(now - buffer_ms, timeframe) - step


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def candle_id(symbol: str, timeframe: str, open_ms: int) -> str:
    """Deterministic identity for one (symbol, timeframe, candle). This is the
    backbone of duplicate-entry protection: the same candle can never be acted
    on twice, even across process restarts."""
    return f"{symbol}|{timeframe}|{open_ms}"
