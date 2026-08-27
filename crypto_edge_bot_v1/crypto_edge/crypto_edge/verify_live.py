"""Live-network verification harness.

Everything in this project is verified offline against deterministic fixtures.
Three things cannot be: the exchange adapter, the market-cap universe provider,
and Telegram delivery. This module exercises exactly those, against the real
network, and prints a report in the shape an operator needs before letting the
bot run continuously.

    python -m crypto_edge.cli verify-live

It is READ-ONLY with respect to trading. It opens no positions and, unless you
pass --cycle, does not run the engine at all. Credentials are read through the
existing .env mechanism and are never printed: `_mask()` guards every place a
token could otherwise reach the terminal.

Exit code 0 means every CRITICAL step passed.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from .config import Config
from .data.feed import DataUnavailable
from .models import TS_LOCAL, TS_VENUE
from .timeutils import candle_close_ms, iso, now_ms, tf_ms

CRITICAL, INFO = "CRITICAL", "INFO"
NEWLINE = chr(10)


def _mask(secret: str) -> str:
    """Never print a credential. Enough to identify it, not enough to use it."""
    if not secret:
        return "<unset>"
    return f"<set, {len(secret)} chars, ...{secret[-4:]}>" if len(secret) > 8 else "<set>"


@dataclass
class Step:
    name: str
    ok: bool
    severity: str
    detail: str = ""


@dataclass
class VerifyReport:
    steps: list[Step] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str = "",
            severity: str = CRITICAL) -> bool:
        self.steps.append(Step(name, ok, severity, detail))
        mark = "PASS" if ok else ("FAIL" if severity == CRITICAL else "WARN")
        print(f"  [{mark:4}] {name:38} {detail}")
        return ok

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.steps if s.severity == CRITICAL)

    def render_summary(self) -> str:
        f = self.facts
        lines = [
            "", "=" * 72, "LIVE VERIFICATION SUMMARY", "=" * 72,
            f"EXCHANGE USED                  {f.get('exchange', '?')}",
            f"LIVE CCXT RESULT               {f.get('ccxt_result', 'NOT RUN')}",
            f"BROAD UNIVERSE RESULT          {f.get('broad_result', 'NOT RUN')}",
            f"NUMBER OF ASSETS RETURNED      {f.get('broad_n', '-')}",
            f"NUMBER INTERSECTING EXCHANGE   {f.get('intersecting', '-')}",
            f"NUMBER AFTER FILTERING         {f.get('after_filter', '-')}",
            f"BTC 1H DATA STATUS             {f.get('btc_1h', '-')}",
            f"BTC 4H DATA STATUS             {f.get('btc_4h', '-')}",
            f"QUOTE/SPREAD STATUS            {f.get('quote_status', '-')}",
            f"QUOTE TIMESTAMP FINDINGS       {f.get('ts_findings', '-')}",
            f"TELEGRAM DELIVERY RESULT       {f.get('telegram', 'NOT RUN')}",
            f"COMPLETE ENGINE CYCLE RESULT   {f.get('cycle', 'NOT RUN')}",
            "=" * 72,
            "VERDICT: " + ("ALL CRITICAL CHECKS PASSED" if self.passed
                           else "FAILURES ABOVE -- do not run continuously yet"),
            "=" * 72,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------- exchange
def verify_exchange(cfg: Config, feed, rep: VerifyReport) -> dict:
    """Markets, metadata, candles, unclosed-candle discard, clock."""
    print(f"\n[1] EXCHANGE ADAPTER -- {cfg.exchange_label()} "
          f"(from {cfg.exchange_source})")
    feed_name = getattr(feed, "name", "")
    if feed_name and feed_name != "fixture":
        rep.add("feed matches configured exchange",
                feed_name == cfg.exchange.name,
                f"feed is '{feed_name}', config says '{cfg.exchange.name}'"
                if feed_name != cfg.exchange.name else f"both are '{feed_name}'")
    out: dict = {}
    rep.facts["exchange"] = f"{cfg.exchange_label()} (from {cfg.exchange_source})"

    try:
        markets = feed.load_markets()
    except DataUnavailable as e:
        rep.add("load_markets", False, str(e))
        rep.facts["ccxt_result"] = f"FAILED: {e}"
        return out
    out["markets"] = markets
    mode = getattr(feed, "precision_mode", "?")
    rep.add("load_markets", len(markets) > 0,
            f"{len(markets)} spot {cfg.exchange.quote} markets, precisionMode={mode}")

    # --- precision / min-notional metadata (the TICK_SIZE bug lived here) ---
    btc = cfg.strategy.btc_symbol
    meta = markets.get(btc)
    if meta is None:
        rep.add("BTC market metadata", False, f"{btc} not listed on this venue")
    else:
        stepped = meta.amount_step > 0 or meta.price_step > 0
        rep.add("exchange precision metadata", True,
                f"{btc} amount_step={meta.amount_step or '-'} "
                f"price_step={meta.price_step or '-'} "
                f"amount_dp={meta.amount_precision} price_dp={meta.price_precision}")
        rep.add("min amount / min notional", True,
                f"min_amount={meta.min_amount} min_cost={meta.min_cost}")
        if not stepped:
            rep.add("tick sizes resolved", True,
                    "venue reports decimal places, not ticks (acceptable)",
                    severity=INFO)
        else:
            sane = meta.amount_precision <= 12 and meta.price_precision <= 12
            rep.add("tick sizes resolved", sane,
                    "ticks converted to decimals correctly" if sane
                    else "implausible decimal count derived from tick")

    # --- how many markets carry usable metadata at all ---
    missing = [s for s, m in markets.items()
               if m.amount_step <= 0 and m.amount_precision >= 8]
    rep.add("markets with granularity data", True,
            f"{len(markets) - len(missing)}/{len(markets)} have explicit "
            f"amount granularity", severity=INFO)

    # --- tickers ---
    try:
        tickers = feed.fetch_tickers()
        out["tickers"] = tickers
        rep.add("fetch_tickers", len(tickers) > 0, f"{len(tickers)} tickers")
    except DataUnavailable as e:
        out["tickers"] = {}
        rep.add("fetch_tickers", False, str(e))

    # --- candles, both timeframes ---
    for label, tf in (("btc_1h", cfg.strategy.entry_timeframe),
                      ("btc_4h", cfg.strategy.regime_timeframe)):
        try:
            raw = feed.fetch_ohlcv(btc, tf, cfg.exchange.ohlcv_limit)
        except DataUnavailable as e:
            rep.add(f"fetch_ohlcv {btc} {tf}", False, str(e))
            rep.facts[label] = f"FAILED: {e}"
            continue
        buffer_ms = cfg.safety.candle_close_buffer_s * 1000
        closed = raw.drop_unclosed(now_ms(), buffer_ms)
        dropped = len(raw) - len(closed)
        if len(closed) == 0:
            rep.add(f"fetch_ohlcv {btc} {tf}", False, "every candle was unclosed")
            rep.facts[label] = "FAILED: no closed candles"
            continue

        last_open = int(closed.open_ms[-1])
        close_ms = candle_close_ms(last_open, tf)
        age_min = (now_ms() - close_ms) / 60_000.0
        # the newest CLOSED candle must genuinely be closed, and no more than
        # one timeframe old (plus the buffer) or we are behind the market
        properly_closed = close_ms + buffer_ms <= now_ms()
        fresh = age_min <= (tf_ms(tf) / 60_000.0) + 5
        sane_ts = bool(closed.is_sane())

        rep.add(f"fetch_ohlcv {btc} {tf}", True,
                f"{len(raw)} rows, {dropped} unclosed discarded, "
                f"last close {iso(close_ms)} ({age_min:.1f} min ago)")
        rep.add(f"unclosed candles discarded ({tf})", properly_closed,
                "newest retained candle is genuinely closed" if properly_closed
                else "a retained candle has NOT closed yet -- look-ahead risk")
        rep.add(f"timestamps sane ({tf})", sane_ts,
                "monotonic, evenly spaced, OHLC consistent" if sane_ts
                else "series failed sanity check")
        rep.add(f"data freshness ({tf})", fresh,
                f"{age_min:.1f} min behind" if fresh
                else f"{age_min:.1f} min behind -- stale for a {tf} strategy",
                severity=CRITICAL if not fresh else INFO)
        rep.facts[label] = (f"OK ({len(raw)} rows, {dropped} unclosed dropped, "
                            f"last close {age_min:.1f} min ago)")
        out[label] = closed

    # --- clock ---
    server = feed.server_time_ms()
    if server is None:
        rep.add("clock synchronisation", True,
                "venue exposes no server time", severity=INFO)
    else:
        skew = abs(server - now_ms()) / 1000.0
        rep.add("clock synchronisation", skew <= cfg.safety.max_clock_skew_s,
                f"local clock differs from venue by {skew:.1f}s")

    rep.facts["ccxt_result"] = "OK" if rep.passed else "FAILURES (see above)"
    return out


# ------------------------------------------------------------------ quotes
def verify_quotes(cfg: Config, feed, broker, samples: int, interval_s: float,
                  rep: VerifyReport) -> None:
    """Spread maths plus the quote-age calibration the policy depends on."""
    print(f"\n[2] QUOTES AND TIMESTAMP CALIBRATION ({samples} samples)")
    btc = cfg.strategy.btc_symbol
    ages: list[float] = []
    spreads: list[float] = []
    venue_stamped = local_stamped = 0
    first = None

    for i in range(max(1, samples)):
        q = feed.fetch_quote(btc)
        if q is None:
            rep.add("fetch_quote", False, f"{btc} returned no quote")
            return
        if first is None:
            first = q
        source = getattr(q, "ts_source", TS_VENUE)
        if source == TS_VENUE:
            venue_stamped += 1
            ages.append((now_ms() - q.ts_ms) / 1000.0)
        else:
            local_stamped += 1
        spreads.append(q.spread_bps)
        if i < samples - 1 and interval_s > 0:
            time.sleep(interval_s)

    rep.add("fetch_quote", True,
            f"bid={first.bid:.8g} ask={first.ask:.8g} last={first.last:.8g}")

    spread_ok = all(s >= 0 for s in spreads)
    rep.add("spread calculation", spread_ok,
            f"median {statistics.median(spreads):.2f} bps, "
            f"max {max(spreads):.2f} bps, entry limit "
            f"{cfg.execution.max_spread_bps_entry:.0f} bps")

    # ---- timestamp findings -------------------------------------------
    if venue_stamped:
        typical = statistics.median(ages)
        worst = max(ages)
        rep.add("venue quote timestamps", True,
                f"{venue_stamped}/{samples} venue-stamped; "
                f"median age {typical:.2f}s, max {worst:.2f}s")
        headroom = cfg.execution.max_quote_age_s / max(worst, 0.001)
        within = worst < cfg.execution.max_quote_age_s
        rep.add("max_quote_age_s is appropriate", within,
                f"max observed {worst:.2f}s vs limit "
                f"{cfg.execution.max_quote_age_s}s ({headroom:.0f}x headroom)"
                if within else
                f"max observed {worst:.2f}s EXCEEDS limit "
                f"{cfg.execution.max_quote_age_s}s -- investigate before raising it")
        rep.facts["ts_findings"] = (
            f"venue-stamped {venue_stamped}/{samples}; median age {typical:.2f}s, "
            f"max {worst:.2f}s; limit {cfg.execution.max_quote_age_s}s "
            f"({'OK' if within else 'TOO TIGHT OR VENUE LAGGING'})")
    else:
        rep.add("venue quote timestamps", True,
                f"venue supplies NO ticker timestamp ({local_stamped}/{samples} "
                f"stamped locally) -- age check is vacuous here and is SKIPPED; "
                f"price-deviation check carries the load", severity=INFO)
        rep.facts["ts_findings"] = (
            f"venue supplies no ticker timestamp; all {local_stamped} samples "
            f"stamped locally. Age check correctly skipped (not silently passed). "
            f"Rely on max_quote_deviation_pct; do NOT set "
            f"require_venue_quote_timestamp on this venue.")

    # ---- run the real entry gate on a real quote -----------------------
    ref = float(first.last)
    qc = broker.validate_entry_quote(
        first, ref_price=ref, max_age_s=cfg.execution.max_quote_age_s,
        max_future_skew_s=cfg.execution.max_quote_future_skew_s,
        max_deviation_pct=cfg.execution.max_quote_deviation_pct,
        require_venue_timestamp=cfg.execution.require_venue_quote_timestamp)
    rep.add("entry quote gate accepts a live quote", qc.ok,
            qc.reason or f"spread {qc.spread_bps:.2f}bps, age {qc.age_s:.2f}s, "
                         f"source={qc.ts_source}, age_verified={qc.age_verified}")
    rep.facts["quote_status"] = (
        f"OK -- median spread {statistics.median(spreads):.2f}bps, "
        f"gate {'accepts' if qc.ok else 'REJECTS'} live quotes"
        + ("" if qc.ok else f" ({qc.reason})"))


# ------------------------------------------------------------- universe
def verify_universe(cfg: Config, repo, broad_service, markets, tickers,
                    rep: VerifyReport) -> None:
    from .data.universe import UniverseBuilder

    print("\n[3] BROAD ASSET UNIVERSE")
    before = repo.latest_broad_universe()
    broad = broad_service.get(force=True)
    if broad is None:
        rep.add("broad universe fetch", False,
                broad_service.last_error or "provider returned nothing")
        rep.facts["broad_result"] = f"FAILED: {broad_service.last_error}"
        # a cold start with no provider AND no cache must fail closed
        rep.add("fails closed with no universe", before is None,
                "no cache either -- entries would be correctly suspended"
                if before is None else "a cache exists; entries would continue",
                severity=INFO)
        return

    p = broad.provenance()
    expected = cfg.universe.broad_limit
    close_enough = len(broad) >= min(expected, cfg.universe.broad_min_assets)
    rep.add("broad universe fetch", close_enough,
            f"{len(broad)} assets (asked for {expected}), scanned "
            f"{p['scanned']} for collisions, source={p['source']}")
    rep.facts["broad_result"] = f"OK ({p['source']})"
    rep.facts["broad_n"] = len(broad)

    # --- cache written, with provenance ---
    row = repo.latest_broad_universe()
    rep.add("cache written", row is not None and row["fetched_ms"] == broad.fetched_ms,
            f"fetched_ms={broad.fetched_ms} ({iso(broad.fetched_ms)})")
    rep.add("source + timestamp stored", bool(row and row["source"] and row["as_of_ms"]),
            f"source={row['source']} as_of={iso(int(row['as_of_ms']))}" if row else "-")
    rep.add("content hash stored",
            bool(row and row["content_hash"] == broad.content_hash),
            f"{broad.content_hash[:16]}...")

    # --- unwanted assets removed ---
    from .data.broad_universe import NON_ASSET_BASES
    leaked = sorted({a.symbol for a in broad.assets} & NON_ASSET_BASES)
    rep.add("stablecoins / wrapped removed", not leaked,
            "none present" if not leaked else f"LEAKED: {leaked}")

    # --- symbol collisions ---
    if broad.ambiguous:
        rep.add("ticker collisions", True,
                f"{len(broad.ambiguous)} ambiguous ticker(s) REFUSED: "
                f"{sorted(broad.ambiguous)[:8]} -- pin them with "
                f"universe.broad_symbol_overrides if you want them traded",
                severity=INFO)
    else:
        rep.add("ticker collisions", True,
                "no ticker in the tradable set is claimed by two assets",
                severity=INFO)

    # --- intersection with the venue ---
    if not markets:
        return
    resolved = [s for s, m in markets.items() if broad.resolve(m.base).ok]
    rep.add("exchange intersection", len(resolved) > 0,
            f"{len(resolved)} of {len(markets)} venue markets map to a "
            f"broad-universe asset")
    rep.facts["intersecting"] = len(resolved)

    builder = UniverseBuilder(cfg.universe)
    keep, audit = builder.build_candidates(markets, tickers, broad)
    rep.add("liquidity / spread / static filters", len(keep) > 0,
            f"{len(keep)} candidates survive stage 1 "
            f"(cap {cfg.universe.max_tradable} enter the pipeline)")
    rep.facts["after_filter"] = len(keep)

    reasons: dict[str, int] = {}
    for r in audit:
        if not r["included"] and r["reject_reason"]:
            key = r["reject_reason"].split("(")[0].split("$")[0].strip()[:52]
            reasons[key] = reasons.get(key, 0) + 1
    print("        top rejection reasons:")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"          {n:5}  {reason}")

    # --- outage fallback, exercised for real ---
    class _Dead:
        name = "simulated_outage"

        def fetch(self, limit):
            raise RuntimeError("simulated provider outage")

    real, broad_service.provider = broad_service.provider, _Dead()
    try:
        fallback = broad_service.get(force=True)
        rep.add("provider outage falls back to cache",
                fallback is not None and fallback.from_cache,
                f"served {len(fallback)} cached assets, stale={fallback.stale}"
                if fallback else "NO fallback -- entries would be suspended")
        if fallback is not None:
            rep.add("cached universe keeps collision data",
                    fallback.ambiguous == broad.ambiguous,
                    f"{len(fallback.ambiguous)} ambiguous ticker(s) preserved",
                    severity=INFO)
    finally:
        broad_service.provider = real


# ------------------------------------------------------------- telegram
def verify_telegram(cfg: Config, repo, notifier, rep: VerifyReport) -> None:
    print("\n[4] TELEGRAM REAL DELIVERY")
    print(f"        token {_mask(cfg.telegram_token)}  "
          f"chat_id {_mask(cfg.telegram_chat_id)}")
    if not cfg.telegram.enabled:
        rep.add("telegram enabled", True, "disabled in config -- skipping",
                severity=INFO)
        rep.facts["telegram"] = "SKIPPED (disabled)"
        return
    if not notifier.enabled:
        rep.add("telegram credentials", False,
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing from .env")
        rep.facts["telegram"] = "FAILED (no credentials)"
        return

    ok, detail = notifier.test_connectivity()
    rep.add("connectivity / self-check message", ok, detail[:120])
    if not ok:
        rep.facts["telegram"] = f"FAILED: {detail[:80]}"
        return

    from .notify import formatters as fmt
    started = notifier.send(
        fmt.bot_start(mode=cfg.safety.mode, exchange=cfg.exchange_label(),
                      equity=cfg.execution.starting_equity,
                      cash=cfg.execution.starting_equity, open_positions=0,
                      strategy=cfg.strategy.name, version=cfg.strategy.version,
                      universe_size=0, telegram_ok=True),
        kind="bot_start")
    rep.add("bot-start message delivered", started, "sent to chat")

    # ---- durable outbox, end to end, against the real transport ----------
    key = f"verify:{now_ms()}"
    delivered = notifier.send("🔍 Crypto Edge verify-live: outbox test message",
                              dedupe_key=key, kind="verify")
    row = repo.telegram_status(key)
    rep.add("outbox row reaches SENT", delivered and row and row["status"] == "SENT",
            f"status={row['status'] if row else 'missing'} "
            f"attempts={row['attempts'] if row else '-'}")

    before = row["sent_ms"] if row else 0
    again = notifier.send("🔍 duplicate that must not be delivered",
                          dedupe_key=key, kind="verify")
    row2 = repo.telegram_status(key)
    suppressed = again and row2 and row2["sent_ms"] == before
    rep.add("duplicate dedupe key suppressed", bool(suppressed),
            "second send did not reach Telegram" if suppressed
            else "DUPLICATE WAS DELIVERED")

    counts = repo.telegram_outbox_counts()
    rep.add("outbox has no abandoned messages", counts.get("FAILED", 0) == 0,
            f"{counts}", severity=INFO)
    rep.facts["telegram"] = "OK (connectivity, bot-start, outbox PENDING->SENT, dedupe)"


def verify_pending_first(notifier, repo, rep: VerifyReport) -> None:
    """Prove a row is PENDING *before* delivery, using a deliberately dead
    transport, then recover it through the real one."""
    print("\n[5] OUTBOX STATE MACHINE (PENDING before SENT)")

    class _Dead:
        def send(self, token, chat_id, text, timeout):
            return False, "deliberate failure for verification"

    real, notifier.transport = notifier.transport, _Dead()
    lease, notifier.outbox_lease_ms = notifier.outbox_lease_ms, 0
    key = f"verify-pending:{now_ms()}"
    try:
        notifier.send("🔍 Crypto Edge verify-live: recovery test",
                      dedupe_key=key, kind="verify")
        row = repo.telegram_status(key)
        rep.add("failed send stays PENDING", bool(row and row["status"] == "PENDING"),
                f"status={row['status'] if row else 'missing'}, "
                f"payload stored={bool(row and row['text'])}")
    finally:
        notifier.transport = real

    recovered = notifier.flush_pending()
    row = repo.telegram_status(key)
    rep.add("pending message recovers to SENT",
            recovered >= 1 and row and row["status"] == "SENT",
            f"flush delivered {recovered}, status={row['status'] if row else '-'}")
    notifier.outbox_lease_ms = lease


# ------------------------------------------------------------- diagnostics
def diagnose_universe(cfg: Config, repo, broad_service, feed,
                      limit: int | None = None) -> dict:
    """Account for EVERY venue market: where it was dropped, and why.

    Written because "the engine is only evaluating two symbols" is not a
    debuggable statement. Every market the venue lists is followed through the
    whole pipeline and lands in exactly one bucket, and the buckets sum to the
    total -- so nothing can go missing without showing up here.

    Stage 2 costs one OHLCV fetch per surviving symbol, so it is opt-in via
    `limit` when the survivor list is long.
    """
    from .data.market_age import MarketAgeService
    from .data.universe import UniverseBuilder

    print("=" * 72)
    print(f"UNIVERSE DIAGNOSTIC -- {cfg.exchange_label()}")
    print("=" * 72)
    reach_days = (cfg.universe.age_probe_bars
                  * tf_ms(cfg.universe.age_probe_timeframe) / 86_400_000)
    rows = [
        (f"required {cfg.strategy.entry_timeframe} history (indicators only)",
         f"{cfg.required_history_bars(cfg.strategy.entry_timeframe)} bars"),
        (f"required {cfg.strategy.regime_timeframe} history (indicators only)",
         f"{cfg.required_history_bars(cfg.strategy.regime_timeframe)} bars"),
        ("market-age method",
         f"{cfg.universe.age_probe_bars} x {cfg.universe.age_probe_timeframe} probe "
         f"({reach_days:.0f} days of reach) vs a "
         f"{cfg.universe.min_market_age_days}-day gate"),
        ("24h volume floor",
         f"{cfg.universe.min_dollar_volume_24h:,.0f} {cfg.quote_currency} of "
         f"notional traded"),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label.ljust(width)} : {value}")

    markets = feed.load_markets()
    tickers = feed.fetch_tickers()
    broad = broad_service.get()
    if broad is None:
        print("NO BROAD UNIVERSE -- entries are suspended; nothing to diagnose.")
        return {"error": "no broad universe"}

    print(f"venue markets ({cfg.quote_currency} spot)              : {len(markets)}")
    print(f"broad universe assets                     : {len(broad)} "
          f"({broad.source}, scanned {broad.scanned})")

    # ---- stage 0: intersection -------------------------------------------
    intersect, not_in_broad, ambiguous = [], [], []
    for sym, meta in markets.items():
        res = broad.resolve(meta.base)
        if res.ok:
            intersect.append(sym)
        elif "claimed by" in res.reason:
            ambiguous.append((sym, res.reason))
        else:
            not_in_broad.append(sym)
    print(f"intersecting the broad universe           : {len(intersect)}")

    # ---- the 24h volume audit, every intersecting market, raw fields -----
    print(NEWLINE + "--- 24H VOLUME AUDIT (all intersecting markets) " + "-" * 24)
    print(f"    threshold: {cfg.universe.min_dollar_volume_24h:,.0f} "
          f"{cfg.quote_currency} of 24h notional (in the SELECTED quote currency)")
    print(f"    {'symbol':16} {'baseVolume':>16} {'quoteVolume':>16} "
          f"{'vwap':>12} {'last':>12} {('-> ' + cfg.quote_currency):>16}  "
          f"{'via':<18} R")
    vol_rows = []
    for sym in sorted(intersect):
        t = tickers.get(sym) or {}
        notional, how = UniverseBuilder.quote_volume(t)
        ok = notional >= cfg.universe.min_dollar_volume_24h
        vol_rows.append((sym, t, notional, how, ok))
        print(f"    {sym:16} {_f(t.get('baseVolume')):>16} "
              f"{_f(t.get('quoteVolume')):>16} {_f(t.get('vwap')):>12} "
              f"{_f(t.get('last')):>12} {notional:>16,.0f}  {how:<18} "
              f"{'PASS' if ok else 'FAIL'}")
    passing = sum(1 for r in vol_rows if r[4])
    print(f"    {passing}/{len(vol_rows)} clear the threshold")
    unavailable = [r[0] for r in vol_rows if r[3] == "unavailable"]
    if unavailable:
        print(f"    !! {len(unavailable)} market(s) published NO usable volume "
              f"field: {unavailable[:8]}")
        print("    !! that is an adapter/venue problem, not illiquidity")

    # ---- stage 1: static / liquidity / spread ----------------------------
    keep, audit = UniverseBuilder(cfg.universe).build_candidates(
        markets, tickers, broad)
    stage1_reasons: dict[str, list] = {}
    for row in audit:
        if row["included"]:
            continue
        reason = _bucket(row["reject_reason"])
        stage1_reasons.setdefault(reason, []).append(row["symbol"])

    tradable = keep[:cfg.universe.max_tradable]
    over_cap = keep[cfg.universe.max_tradable:]

    print(f"surviving stage 1 (static/liquidity/spread): {len(keep)}")
    print(f"entering the pipeline (max_tradable={cfg.universe.max_tradable:<3})   : "
          f"{len(tradable)}")

    print(NEWLINE + "--- STAGE 1 REJECTIONS BY REASON " + "-" * 38)
    for reason, syms in sorted(stage1_reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(syms):5}  {reason}")
        print(f"         e.g. {', '.join(sorted(syms)[:6])}")

    # ---- stage 2: history / age / volatility -----------------------------
    probe = tradable if limit is None else tradable[:limit]
    print(NEWLINE + f"--- STAGE 2 (per-symbol history fetch) on {len(probe)} "
          f"symbol(s) " + "-" * 12)
    print("    this is where a venue history cap shows up" + NEWLINE)

    builder = UniverseBuilder(cfg.universe)
    age_svc = MarketAgeService(
        repo, probe_timeframe=cfg.universe.age_probe_timeframe,
        probe_bars=cfg.universe.age_probe_bars,
        cache_hours=cfg.universe.age_cache_hours)
    stage2_reasons: dict[str, list] = {}
    survivors, bar_counts, atr_only = [], [], []
    want_1h = cfg.required_history_bars(cfg.strategy.entry_timeframe)
    want_htf = cfg.required_history_bars(cfg.strategy.regime_timeframe)
    print(f"    requesting {want_1h} x {cfg.strategy.entry_timeframe} and "
          f"{want_htf} x {cfg.strategy.regime_timeframe} per symbol")

    for sym in probe:
        try:
            raw = feed.fetch_ohlcv(sym, cfg.strategy.entry_timeframe, want_1h)
        except DataUnavailable as e:
            stage2_reasons.setdefault("1h fetch failed", []).append(sym)
            print(f"    {sym:16} FETCH FAILED  {str(e)[:60]}")
            continue
        series = raw.drop_unclosed(now_ms(), cfg.safety.candle_close_buffer_s * 1000)
        bars = len(series)
        bar_counts.append((sym, bars))
        span_d = ((int(series.open_ms[-1]) - int(series.open_ms[0])) / 86_400_000
                  if bars > 1 else 0.0)

        try:
            htf = feed.fetch_ohlcv(sym, cfg.strategy.regime_timeframe, want_htf)
            htf_bars = len(htf.drop_unclosed(
                now_ms(), cfg.safety.candle_close_buffer_s * 1000))
        except DataUnavailable:
            htf_bars = 0

        atr_pct = _atr_pct(series, cfg.strategy.atr_period)
        reason = builder.filter_by_history(sym, series, atr_pct,
                                           requested_bars=want_1h)
        if not reason and htf_bars < cfg.strategy.regime_ema + 2:
            reason = (f"insufficient {cfg.strategy.regime_timeframe} history "
                      f"({htf_bars} bars)")
        verdict = age_svc.age_of(sym, meta=markets.get(sym), feed=feed)
        if not reason and cfg.universe.min_market_age_days > 0:
            reason = verdict.reason_if_blocked(cfg.universe.min_market_age_days)
        age_txt = ("?" if not verdict.known
                   else f"{verdict.age_days:.0f}d/{verdict.source[:12]}")
        print(f"    {sym:16} 1h={bars:<5} {cfg.strategy.regime_timeframe}={htf_bars:<5} "
              f"span={span_d:6.1f}d age={age_txt:<22} "
              f"atr={atr_pct if atr_pct is None else round(atr_pct, 2)}"
              f"  {reason or 'ELIGIBLE'}")
        if reason:
            stage2_reasons.setdefault(_bucket(reason), []).append(sym)
            # Would this market be eligible if the ATR floor alone were lifted?
            # Recorded so the threshold can be evaluated against data later
            # rather than adjusted during an infrastructure fix.
            if reason.startswith("too quiet"):
                without_atr = builder.filter_by_history(sym, series, None,
                                                        requested_bars=want_1h)
                age_ok = (cfg.universe.min_market_age_days <= 0
                          or not verdict.reason_if_blocked(
                              cfg.universe.min_market_age_days))
                if not without_atr and age_ok:
                    atr_only.append((sym, atr_pct))
        else:
            survivors.append(sym)

    # ---- the accounting ---------------------------------------------------
    print(NEWLINE + "=" * 72)
    print("DIAGNOSTIC SUMMARY -- every venue market accounted for")
    print("=" * 72)
    print(f"  {len(markets):5}  venue {cfg.exchange.quote} spot markets")
    print(f"  {len(not_in_broad):5}  not in the broad top-N asset universe")
    print(f"  {len(ambiguous):5}  ambiguous ticker (claimed by >1 asset)")
    for reason, syms in sorted(stage1_reasons.items(), key=lambda kv: -len(kv[1])):
        if "broad top-N" in reason or "claimed by" in reason:
            continue
        print(f"  {len(syms):5}  stage 1: {reason}")
    print(f"  {len(over_cap):5}  beyond max_tradable cap")
    for reason, syms in sorted(stage2_reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(syms):5}  stage 2: {reason}")
    print(f"  {len(survivors):5}  ELIGIBLE FOR SIGNAL EVALUATION")
    if atr_only:
        print(NEWLINE + f"  ATR FLOOR OBSERVATION: {len(atr_only)} market(s) pass "
              f"EVERY other gate and fail only the {cfg.universe.min_atr_pct}% "
              f"ATR minimum.")
        print("  (reported, not acted on -- evaluate the threshold with data)")
        for sym, a in sorted(atr_only, key=lambda x: -(x[1] or 0))[:15]:
            print(f"      {sym:16} ATR {a:.3f}%")
    if survivors:
        print(f"         {', '.join(survivors)}")

    if bar_counts:
        counts = [b for _, b in bar_counts]
        # Compare against the REQUIREMENT, not the padded request. The request
        # carries a small margin for venue rounding, so "one bar short of the
        # padded ask" is normal and must not be reported as a venue cap.
        floor = cfg.universe.min_candles_1h
        short = [(s, b) for s, b in bar_counts if b < floor]
        print(NEWLINE + f"  1h bars returned: min={min(counts)} max={max(counts)} "
              f"requested={want_1h}")
        if short:
            print(f"  {len(short)}/{len(bar_counts)} symbols came back below the "
                  f"{floor}-bar requirement -- the venue is capping history.")
            print(f"  Worst: {sorted(short, key=lambda x: x[1])[:5]}")
        else:
            print(f"  every symbol met the {floor}-bar indicator requirement")
    print("=" * 72)

    return {"markets": len(markets), "intersecting": len(intersect),
            "stage1_kept": len(keep), "tradable": len(tradable),
            "eligible": len(survivors), "atr_only": atr_only,
            "stage1_reasons": stage1_reasons,
            "stage2_reasons": stage2_reasons, "bar_counts": bar_counts,
            "requested_1h": want_1h}


def _f(v) -> str:
    """Format a raw ticker field for the audit table, preserving its absence."""
    if v is None:
        return "None"
    try:
        return f"{float(v):,.4g}"
    except (TypeError, ValueError):
        return repr(v)[:16]


def _bucket(reason: str) -> str:
    """Collapse a reason to its family so counts are meaningful."""
    r = (reason or "").strip()
    for prefix in ("venue supplied only", "history truncated by venue",
                   "market too new", "only ", "24h volume", "spread ",
                   "too quiet", "too volatile", "not in the broad",
                   "outside top", "candle data failed", "no candle data",
                   "market age could not be established", "insufficient "):
        if r.startswith(prefix):
            return prefix.strip() if prefix.strip() != "only" else "insufficient 1h history"
    if "claimed by" in r:
        return "ambiguous ticker"
    return r.split("(")[0].split("$")[0].strip()[:56] or "unknown"


def _atr_pct(series, period: int):
    import numpy as np

    from .indicators import atr, last_valid
    if len(series) <= period + 2:
        return None
    a = last_valid(atr(series.high, series.low, series.close, period))
    px = float(series.close[-1])
    if not np.isfinite(a) or px <= 0:
        return None
    return a / px * 100.0


# ------------------------------------------------------------------ cycle
def verify_cycle(cfg: Config, repo, feed, notifier, rep: VerifyReport) -> None:
    """One complete engine cycle on live data. Zero entries is a pass."""
    from .engine import TradingEngine

    print("\n[6] ONE COMPLETE LIVE-DATA PAPER CYCLE")
    engine = TradingEngine(cfg, repo, feed, notifier)
    before_obs = len(repo.get_observations())
    t0 = time.time()
    try:
        engine.cycle()
    except Exception as e:
        rep.add("engine cycle completes", False, f"{type(e).__name__}: {e}")
        rep.facts["cycle"] = f"FAILED: {e}"
        return
    elapsed = time.time() - t0
    st = engine.status

    rep.add("engine cycle completes", True, f"{elapsed:.1f}s")
    rep.add("universe resolved", bool(st.universe) or not st.broad_universe_ok,
            f"{len(st.universe)} tradable, broad={st.broad_universe_n} "
            f"({st.broad_universe_source}), stale={st.broad_universe_stale}")
    rep.add("BTC context built", st.btc_regime != "unknown",
            f"regime={st.btc_regime} score={st.btc_regime_score:.0f} "
            f"breadth={st.breadth:.0f}% environment={st.environment}",
            severity=INFO)
    rep.add("HTF context built",
            cfg.strategy.btc_symbol in engine._series_htf,
            f"{len(engine._series_htf)} symbols with "
            f"{cfg.strategy.regime_timeframe} series")
    rep.add("signals evaluated", True,
            f"{st.signals_evaluated} evaluated, {st.entries} entries, "
            f"{st.exits} exits", severity=INFO)
    rep.add("open positions managed", True,
            f"{len(engine.account.positions())} open", severity=INFO)
    rep.add("circuit breakers evaluated", True,
            f"halted={st.halted} {st.halt_reason}", severity=INFO)

    after_obs = repo.get_observations()
    written = len(after_obs) - before_obs
    rep.add("journal writes", written >= 0,
            f"{written} new observations this cycle", severity=INFO)
    if after_obs:
        decisions: dict[str, int] = {}
        for o in after_obs:
            decisions[o["decision"]] = decisions.get(o["decision"], 0) + 1
        print(f"        decisions to date: {decisions}")
        rejects = [o for o in after_obs if o["reject_reason"]]
        if rejects:
            print(f"        sample rejection: {rejects[-1]['symbol']} -- "
                  f"{rejects[-1]['reject_reason'][:80]}")

    rep.add("data errors during cycle", st.data_errors == 0,
            f"{st.data_errors} symbol fetch failures",
            severity=INFO)
    rep.facts["cycle"] = (f"OK -- {st.signals_evaluated} signals evaluated, "
                          f"{st.entries} entries, {len(st.universe)} in universe, "
                          f"{elapsed:.1f}s")
