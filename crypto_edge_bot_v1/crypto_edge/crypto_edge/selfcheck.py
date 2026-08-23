"""Startup self-check.

The engine refuses to mark itself PAPER TRADING ACTIVE until every CRITICAL
check passes. Failures are explained precisely rather than logged vaguely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .logging_setup import log_event
from .storage import db
from .storage.repo import Repo
from .timeutils import iso, now_ms, tf_ms

CRITICAL, WARNING = "CRITICAL", "WARNING"


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str
    detail: str = ""


@dataclass
class SelfCheckReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, severity: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, ok, severity, detail))

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results if r.severity == CRITICAL)

    def render(self) -> str:
        lines = ["STARTUP SELF-CHECK", "=" * 60]
        for r in self.results:
            mark = "PASS" if r.ok else ("FAIL" if r.severity == CRITICAL else "WARN")
            lines.append(f"[{mark:4}] {r.name:34} {r.detail}")
        lines.append("=" * 60)
        lines.append("PAPER TRADING ACTIVE" if self.passed
                     else "STOPPED — critical checks failed (see FAIL lines above)")
        return "\n".join(lines)


def run_selfcheck(cfg: Config, repo: Repo | None, feed, notifier,
                  check_network: bool = True) -> SelfCheckReport:
    rep = SelfCheckReport()

    # ---- configuration ---------------------------------------------------
    errs = cfg.validate()
    rep.add("configuration valid", not errs, CRITICAL,
            "; ".join(errs) if errs else "all parameters within bounds")

    # ---- hard paper-mode assertion --------------------------------------
    paper_ok = (cfg.safety.mode.upper() == "PAPER" and not cfg.safety.live_trading_enabled)
    rep.add("PAPER mode enforced", paper_ok, CRITICAL,
            "no real-order code path exists" if paper_ok else "LIVE MODE REQUESTED — refusing")

    # ---- database --------------------------------------------------------
    if repo is None:
        rep.add("database access", False, CRITICAL, "no database connection")
        return rep
    try:
        ok, detail = db.integrity_ok(repo.conn)
        rep.add("database integrity", ok, CRITICAL, detail)
    except Exception as e:
        rep.add("database integrity", False, CRITICAL, str(e))

    try:
        acct = repo.ensure_account(cfg.execution.starting_equity)
        equity_ok = float(acct["cash"]) >= 0
        rep.add("account state", equity_ok, CRITICAL,
                f"cash ${float(acct['cash']):,.2f}, "
                f"start ${float(acct['starting_equity']):,.2f}, "
                f"peak ${float(acct['peak_equity']):,.2f}"
                + (" [HALTED: " + acct["halt_reason"] + "]" if int(acct["halted"]) else ""))
    except Exception as e:
        rep.add("account state", False, CRITICAL, str(e))
        return rep

    try:
        positions = repo.get_positions()
        detail = f"{len(positions)} open position(s)"
        if positions:
            detail += ": " + ", ".join(
                f"{p.symbol} qty={p.qty:g} stop=${p.current_stop:,.6g}" for p in positions[:6])
        rep.add("open positions restored", True, CRITICAL, detail)
    except Exception as e:
        rep.add("open positions restored", False, CRITICAL, str(e))

    # ---- market data (NETWORK) -------------------------------------------
    if not check_network:
        rep.add("market data connection", True, WARNING, "skipped (offline mode)")
        rep.add("exchange symbol metadata", True, WARNING, "skipped (offline mode)")
        rep.add("clock synchronisation", True, WARNING, "skipped (offline mode)")
        rep.add("latest candle freshness", True, WARNING, "skipped (offline mode)")
    else:
        markets = {}
        try:
            markets = feed.load_markets()
            rep.add("exchange symbol metadata", len(markets) > 0, CRITICAL,
                    f"{len(markets)} spot {cfg.exchange.quote} markets loaded")
        except Exception as e:
            rep.add("exchange symbol metadata", False, CRITICAL, str(e))

        btc = cfg.strategy.btc_symbol
        try:
            s = feed.fetch_ohlcv(btc, cfg.strategy.entry_timeframe, 50)
            fresh = len(s) > 0
            detail = (f"{btc}: {len(s)} candles, last open {iso(int(s.open_ms[-1]))}"
                      if fresh else "no candles returned")
            rep.add("market data connection", fresh, CRITICAL, detail)
            if fresh:
                age_s = (now_ms() - int(s.open_ms[-1])) / 1000.0
                tolerable = tf_ms(cfg.strategy.entry_timeframe) / 1000.0 * 2
                rep.add("latest candle freshness", age_s <= tolerable, CRITICAL,
                        f"last candle {age_s / 60:.1f} min old")
        except Exception as e:
            rep.add("market data connection", False, CRITICAL, str(e))

        try:
            server = feed.server_time_ms()
            if server is None:
                rep.add("clock synchronisation", True, WARNING,
                        "exchange does not expose server time")
            else:
                skew = abs(server - now_ms()) / 1000.0
                rep.add("clock synchronisation", skew <= cfg.safety.max_clock_skew_s,
                        CRITICAL, f"local clock differs from exchange by {skew:.1f}s")
        except Exception as e:
            rep.add("clock synchronisation", False, WARNING, str(e))

    # ---- telegram (NETWORK) ---------------------------------------------
    if not cfg.telegram.enabled:
        rep.add("telegram connectivity", True, WARNING, "disabled in config")
    elif not check_network:
        rep.add("telegram connectivity", True, WARNING, "skipped (offline mode)")
    else:
        try:
            ok, detail = notifier.test_connectivity()
            # WARNING, not CRITICAL: losing Telegram must never stop the trader
            rep.add("telegram connectivity", ok, WARNING, detail)
        except Exception as e:
            rep.add("telegram connectivity", False, WARNING, str(e))

    for r in rep.results:
        if r.ok:
            level, verb = "INFO", "passed"
        elif r.severity == CRITICAL:
            level, verb = "ERROR", "FAILED"
        else:
            # A non-critical check that failed is a warning, not an error.
            # Logging a Telegram outage at ERROR alongside database corruption
            # would train whoever reads these logs to ignore both.
            level, verb = "WARNING", "warned"
        log_event("app", level, f"selfcheck {verb}: {r.name}",
                  ok=r.ok, severity=r.severity, detail=r.detail)
    return rep
