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
                  check_network: bool = True,
                  broad_service=None) -> SelfCheckReport:
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
        acct = repo.ensure_account(cfg.strategy.name,
                                   cfg.starting_equity_for(cfg.strategy.name))
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
        positions = repo.get_positions(cfg.strategy.name)
        detail = f"{len(positions)} open position(s)"
        if positions:
            detail += ": " + ", ".join(
                f"{p.symbol} qty={p.qty:g} stop=${p.current_stop:,.6g}" for p in positions[:6])
        rep.add("open positions restored", True, CRITICAL, detail)
    except Exception as e:
        rep.add("open positions restored", False, CRITICAL, str(e))

    # ---- which venue is this, and did it just change? --------------------
    # The override is a per-invocation flag, so forgetting it silently moves the
    # bot to a different exchange. The database carries venue-specific state
    # (positions, processed candles, the research journal), so a change is
    # reported prominently rather than discovered later in the numbers.
    rep.add("effective exchange", True, CRITICAL,
            f"{cfg.exchange_label()}  (from {cfg.exchange_source})")
    try:
        changed, previous = repo.record_exchange(cfg.exchange_label())
        if changed:
            prev_venue, _, prev_quote = previous.partition("/")
            what = ("QUOTE CURRENCY" if prev_venue == cfg.exchange.name
                    else "EXCHANGE")
            extra = ""
            if prev_quote and prev_quote != cfg.quote_currency:
                # Prices, volumes, equity and every stored fill are denominated
                # in the quote currency. Mixing two of them in one account is
                # not a reporting nuisance, it is a wrong P&L.
                extra = (f" Account equity and every recorded fill are "
                         f"denominated in {prev_quote}, not {cfg.quote_currency}.")
            rep.add("exchange unchanged since last run", False, WARNING,
                    f"{what} CHANGED: this database was last used with "
                    f"{previous} and is now {cfg.exchange_label()} -- positions, "
                    f"processed candles and journal rows in it came from "
                    f"{previous}.{extra} Use a separate engine.db_path per "
                    f"venue/quote unless this is deliberate.")
        else:
            rep.add("exchange unchanged since last run", True, WARNING,
                    f"database belongs to {cfg.exchange_label()}")
    except Exception as e:
        rep.add("exchange unchanged since last run", False, WARNING, str(e))

    # ---- undelivered notifications --------------------------------------
    # WARNING, not CRITICAL: a stuck outbox means someone is not being told
    # something, which is serious, but stopping the trader over it would be worse.
    try:
        counts = repo.telegram_outbox_counts()
        pending, failed = counts.get("PENDING", 0), counts.get("FAILED", 0)
        rep.add("notification outbox", failed == 0, WARNING,
                f"{pending} pending, {failed} abandoned"
                + (" -- messages were never delivered" if failed else ""))
    except Exception as e:
        rep.add("notification outbox", False, WARNING, str(e))

    # ---- broad asset universe -------------------------------------------
    # CRITICAL only when entries actually depend on it. Without a universe the
    # bot still runs and still manages open positions, but it opens nothing new.
    if broad_service is None:
        rep.add("broad asset universe", True, WARNING, "not checked")
    else:
        try:
            broad = broad_service.get()
            if broad is None:
                rep.add("broad asset universe",
                        not cfg.universe.require_broad_universe, CRITICAL,
                        "no current or cached universe -- NEW ENTRIES WILL BE "
                        "SUSPENDED (" + (broad_service.last_error or "unavailable") + ")")
            else:
                p = broad.provenance()
                rep.add("broad asset universe", True,
                        WARNING if broad.stale else CRITICAL,
                        f"{p['n_assets']} assets from {p['source']}, "
                        f"{'CACHED ' if p['from_cache'] else ''}"
                        f"{'STALE ' if p['stale'] else ''}"
                        f"age {p['age_s'] / 3600.0:.1f}h, hash {p['content_hash'][:12]}")
        except Exception as e:
            rep.add("broad asset universe",
                    not cfg.universe.require_broad_universe, CRITICAL, str(e))

    # ---- market data (NETWORK) -------------------------------------------
    if not check_network:
        rep.add("market data connection", True, WARNING, "skipped (offline mode)")
        rep.add("exchange symbol metadata", True, WARNING, "skipped (offline mode)")
        rep.add("clock synchronisation", True, WARNING, "skipped (offline mode)")
        rep.add("latest candle freshness", True, WARNING, "skipped (offline mode)")
    else:
        feed_name = getattr(feed, "name", "")
        if feed_name and feed_name != "fixture":
            rep.add("feed matches configured exchange",
                    feed_name == cfg.exchange.name, CRITICAL,
                    f"feed is '{feed_name}', config says '{cfg.exchange.name}'"
                    if feed_name != cfg.exchange.name
                    else f"both are '{feed_name}'")

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
