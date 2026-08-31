# crypto_edge_bot_v1

A paper-trading crypto system: dynamic Top-200 universe scanning, a trend
breakout strategy, ATR-based risk sizing, persistent SQLite state, Telegram
notifications, and a research database that records every decision — including
the trades it *declined* to take.

**This bot cannot place a real order.** There is no order-placement code path,
no API-key handling, and no exchange credentials anywhere in the project. It
reads public market data and simulates fills against it.

---

## Requirements

- Python **3.11 or newer** (the config loader uses `tomllib`)
- Outbound internet access to your chosen exchange's public API
- A Telegram bot token, if you want notifications

## Setup

```bash
cd crypto_edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
```

Getting Telegram credentials: message `@BotFather` on Telegram to create a bot
and get the token. Then message your new bot once, and visit
`https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` to find your chat id.

If you do not want Telegram, set `enabled = false` under `[telegram]` in
`config/config.toml` and leave `.env` empty.

## First run

```bash
python -m crypto_edge.cli selfcheck    # verify everything before trading
python -m crypto_edge.cli start        # begin paper trading
```

`selfcheck` exits non-zero if any critical check fails, so it is safe to use in
a supervisor or a script. `start` runs the self-check itself and refuses to
start if it fails.

To verify the install without touching the network at all:

```bash
python -m crypto_edge.cli selfcheck --offline
python scripts/smoke_test.py           # full lifecycle on synthetic data
python -m crypto_edge.cli test         # the automated suite
```

## Operating it

| Command | What it does |
|---|---|
| `python -m crypto_edge.cli start` | Run the trader in the foreground. Ctrl-C stops it safely. |
| `python -m crypto_edge.cli status` | Account snapshot: equity, cash, exposure, P&L. |
| `python -m crypto_edge.cli positions` | Open positions with stops and unrealized P&L. |
| `python -m crypto_edge.cli performance` | Full metrics report. `--json` for machine output, `--categories` for breakdowns. |
| `python -m crypto_edge.cli research` | Decision database summary, including counterfactuals. |
| `python -m crypto_edge.cli export --out trades.csv` | Export the closed-trade ledger. |
| `python -m crypto_edge.cli resume --yes` | Clear a circuit-breaker halt (deliberate, manual). |
| `python -m crypto_edge.cli test` | Run the automated test suite. |
| `python -m crypto_edge.cli verify-live --cycle` | Verify exchange, universe and Telegram against the real network. |
| `python -m crypto_edge.cli verify-restart` | Prove persisted state survives a restart. |
| `python -m crypto_edge.cli diagnose` | Explain, per venue market, why it is or is not in the tradable universe. |

Background operation:

```bash
scripts/start.sh      # self-check, then run detached with a pid file
scripts/status.sh     # process state + account snapshot
scripts/stop.sh       # SIGTERM; finishes the current cycle, then exits
```

State is committed to SQLite every cycle, so a hard kill loses at most the
cycle in flight. Restarting resumes from exactly where it stopped — open
positions, stops, equity, peak equity and halt state all persist.

---

## How it decides

Each cycle runs in a fixed order, and **stops are processed before entries** so
a position can never be opened on the same tick that another should have been
closed:

```
safety gates → universe refresh → fetch data (discard unclosed candles)
→ build market context → manage open positions → check circuit breakers
→ evaluate signals → rank → risk gate → enter → bookkeeping
```

**Universe.** Rank the venue by 24h dollar volume, keep the top 200, then
aggressively filter: stablecoins, leveraged tokens (`3L`/`3S`/`UP`/`DOWN`),
wrapped and staked duplicates, thin volume, wide spreads, insufficient history,
markets younger than 45 days, and markets outside a sane volatility band. At
most 40 survive to the strategy. Every rejection is written to
`universe_snapshots` with its reason.

**Entry.** Hard filters run first and in order — history, data sanity, no-trade
list, BTC regime, higher-timeframe regime, EMA alignment, Donchian breakout,
ADX, RSI exhaustion, relative volume, extension from the mean. Only what
survives gets scored, across nine weighted components, and only scores ≥ 55
are eligible. The strategy is a pure function; the portfolio manager decides
what actually gets bought.

**Sizing.** Risk budget first (0.5% of equity at the initial stop), then capped
by max position size, remaining exposure room, and affordable cash including
fees — then rounded *down* to the exchange's precision and checked against
minimum notional.

**Exits.** Chandelier trailing stop that only ever ratchets up, breakeven at
1R, trend invalidation, and a time stop at 96 bars. Gaps through the stop fill
at the next candle's open, not at the stop price — because that is what
actually happens.

**Circuit breakers.** A 3% daily loss halts new entries and clears at UTC
midnight. A 20% drawdown is a kill switch that does **not** auto-reset; you
clear it deliberately with `resume --yes`.

## The look-ahead guard

Every piece of external information carries two timestamps:

- `event_time` — when the thing happened in the world
- `observed_at` — when *this system* learned about it

All visibility queries filter on `observed_at`, never `event_time`. A news
event that occurred an hour ago but that you only receive six hours from now is
invisible to the bot until then. This applies to news, token events, funding,
open interest and liquidations. It exists so that when you later backtest
against this database, you cannot accidentally trade on information you did not
have. There are 25 tests dedicated to this in `tests/test_lookahead.py` and
`tests/test_research_intel.py`.

Candle handling follows the same principle: unfinished candles are discarded
the moment they arrive, the Donchian channel excludes the current bar, and
relative volume uses prior bars only.

## Research database

Every evaluation is recorded — entered, rejected by the strategy, rejected by
risk, or ranked out — with the full feature snapshot and the specific reason.
Rejected signals are then tracked as **counterfactuals**: what *would* have
happened at 1, 4, 12, 24, 48 and 168 hours. These are stored separately, always
flagged `hypothetical=1`, and can never create a position or touch account
performance. Reported per rejection reason, and suppressed below 20 samples so
you do not read signal into noise.

## News, token events and derivatives — exact status

Nothing in this section is collecting data. Read the table before assuming any
of it is doing anything for you.

| Piece | Architectural only | Implemented | Tested | Wired to a provider | Actively collecting |
|---|---|---|---|---|---|
| News storage (`news_events`, `event_time`/`observed_at`) | | yes | yes | n/a | **no** |
| News classification (`classify`, severity, source tiers) | | yes | yes | n/a | **no** |
| News look-ahead guard (`news_visible_at`) | | yes | yes | n/a | **no** |
| News → symbol blocking (`ingest` → `no_trade_list`) | | yes | yes | **no** | **no** |
| `NewsProvider` protocol | yes | interface only | — | **no** | **no** |
| `StubNewsProvider` (returns `[]`) | | yes | yes | n/a | **no** |
| Token-event calendar storage + `blocking_event` | | yes | yes | **no** | **no** |
| Derivatives storage (`derivatives` table) | | yes | yes | n/a | **no** |
| Derivatives look-ahead guard (`latest_derivatives`) | | yes | yes | n/a | **no** |
| `DerivativesProvider` protocol | yes | interface only | — | **no** | **no** |
| `CCXTDerivativesProvider` | | written, **never executed** | **no** | **no** | **no** |
| Derivatives → entry decisions | yes | **not implemented** (deliberate) | — | — | — |

Specifically, and to be unambiguous:

- **No provider is constructed anywhere in the running system.** `TradingEngine`
  builds `NewsEngine(repo, [])` and `DerivativesEngine(repo, [])` with empty
  provider lists.
- **No polling loop exists.** `NewsEngine.poll()` and
  `DerivativesEngine.poll()` are implemented and unit-tested but are **never
  called** by the engine cycle. `intel.news_poll_minutes` and
  `intel.derivatives_poll_minutes` are currently read by nothing.
- **The read paths are live but always empty.** `build_context()` calls
  `news.context_for()` / `derivatives.context_for()` (gated on the `*_enabled`
  flags, both `false`), and `evaluate_signals()` calls
  `calendar.blocking_event()` unconditionally. They query real tables that no
  code path ever writes to outside the tests, so they return nothing.
- `CCXTDerivativesProvider` has never made a network call in any environment.

What that means in practice: **a news event cannot currently stop a trade**, and
no funding/OI data is being recorded for later research. The plumbing to change
that is in place and tested; the providers are not.

## Wiring a news provider

To add one, implement `fetch(since_ms) -> list[NewsItem]` and pass it to
`NewsEngine` — note that you must also arrange for `poll()` to be called, which
the engine does not currently do:

```python
class MyProvider:
    name = "my_source"
    def fetch(self, since_ms: int) -> list[NewsItem]:
        ...
```

Set `observed_at` to the moment **your system received the item** — never the
article's publication timestamp. That distinction is the whole point of the
guard. Only severity ≥ 4 from a tier ≤ 2 source will block trading; a rumour on
social media is logged and never acted on.

---

## Verification status

Read this section before trusting anything.

### HOW TO COMPLETE LIVE VERIFICATION

Three things cannot be verified without outbound network access: the exchange
adapter, the market-cap universe provider, and Telegram delivery. One command
exercises all three and prints a report:

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your real token/chat id in it
python -m crypto_edge.cli verify-live --cycle
python -m crypto_edge.cli verify-restart
```

Switching venue never requires editing a file — pass `--exchange` and
`--quote` (or set `CRYPTO_EDGE_EXCHANGE` / `CRYPTO_EDGE_QUOTE`):

```bash
python -m crypto_edge.cli --exchange kraken --quote USD verify-live --cycle
```

**Quote currency is part of the venue, not a detail.** `kraken/USD` and
`kraken/USDT` are different markets with different liquidity: on Kraken the USD
book is the deep one, and no USDT pair cleared the $5M 24h floor in an August
2026 snapshot. `exchange.quote` is the single source of truth — the BTC regime
reference (`BTC/USD` vs `BTC/USDT`), the always-included markets, every symbol
built for a ranked asset, and the units of the 24h volume floor are all derived
from it. Pinning `strategy.btc_symbol` or `universe.always_include` to a
different currency is a startup error, not a silent mismatch.

The 24h volume floor is **notional traded in the selected quote currency**, so
`$5,000,000` means 5,000,000 USD on `kraken/USD` and 5,000,000 USDT on
`kraken/USDT`. Changing quote currency on an existing database prints a loud
warning: the stored equity, fills and history are still denominated in the
*old* currency, and the bot will not silently mix the two.

**Candle freshness is measured from the CLOSE.** CCXT delivers candle *open*
timestamps: a 1h bar stamped `03:00` spans `03:00 -> 04:00` and is only complete
at `04:00`. Every staleness figure the bot reports is `now - close`, so a 1h and
a 4h series that are both current read as equally fresh. The allowance is one
timeframe plus five minutes, which is what "at most one bar behind" means.

**Cycle pacing.** Cycles never overlap and never queue: `run()` is
single-threaded and finishes one cycle before timing the next. If a cycle takes
longer than `poll_seconds` the bot slows down rather than looping back to back —
the realised cadence is `max(poll_seconds, cycle + min_pause_seconds)`, the
overrun is logged, and the heartbeat says how far behind it is.

**Candle caching.** A closed candle never changes, so the feed keeps
`ohlcv_cache_bars` of them per symbol and timeframe and goes back to the venue
only when a new candle has actually closed — not on a timer. Between bar closes
a cycle makes zero OHLCV requests; a bar close costs one request per series. The
candle in progress is never cached, and if the refresh fails the fetch fails
closed rather than serving what it already has.

**Per-strategy ledgers.** Each strategy has its own independent paper
sub-account: its own cash, peak equity, daily loss halt and trade history. One
strategy's open positions can never change another's available cash, which is
what makes two strategies comparable. `--strategy` picks which ledger a report
reads; it defaults to the configured strategy, so every existing command means
what it always did. Databases from before this change migrate automatically —
the existing account becomes the sub-account of the strategy that traded it,
with its history intact, and the pre-migration table is kept as
`account_pre_v4` so the migration can be audited against its source.

To see exactly which markets survive each gate and why:

```bash
python -m crypto_edge.cli --exchange kraken --quote USD diagnose
```

`verify-live` opens no positions and contains no order code. It reports the
exchange's precision/tick metadata, both candle timeframes, whether unfinished
candles were discarded, quote spreads, **measured quote-age statistics**, the
broad universe (count, cache, provenance hash, ticker collisions, exchange
intersection, survivors after filtering), real Telegram delivery through the
durable outbox, and — with `--cycle` — one complete engine cycle. Zero entries
is a passing result. Credentials are masked in all output.

If a check fails, fix it before running continuously; the summary block ends
with an explicit verdict.

### VERIFIED OFFLINE — 685 automated tests, all passing

Exercised against deterministic synthetic data with no network:

- Position sizing, fee and slippage arithmetic, the full P&L identity
  (`net ≡ gross − fees − slippage`), stop fills including gap-through-at-open
- Account persistence: round-trip cash change equals net P&L exactly; open
  positions, stops, equity and peak equity survive a restart; duplicate-entry
  guards survive a restart
- Risk gates, correlation blocking, both circuit breakers and their priority
- Stop ratcheting, breakeven engagement, time stops
- Look-ahead protection, indicator causality, determinism
- Universe filtering and rejection auditing
- News/event visibility, classification, blocking and expiry (the LOGIC only —
  no provider is wired; see the status table above)
- The Telegram outbox state machine: a failed send stays PENDING and
  recoverable, survives a restart, and a delivered one is never repeated
- Entry quote validation failing closed on missing/invalid/stale/malformed
  quotes, while open positions stay manageable without a quote
- Position sizing against the simulated fill, revalidated post-fill against the
  configured risk budget
- The broad market-cap asset universe: intersection, caching with source and
  timestamp, outage fallback, and fail-closed behaviour for new entries
- Research journal, counterfactual isolation, performance metrics and
  small-sample honesty flags
- Telegram message *formatting*, retry, dedupe and fail-safe logic
- The startup self-check gate
- Full end-to-end lifecycle via `scripts/smoke_test.py`

### REQUIRES VERIFICATION ON YOUR MACHINE

These could not be executed here — the build environment had **no outbound
network**, so `ccxt` could not even be installed:

- **Live market data.** `crypto_edge/data/ccxt_feed.py` is written against the
  ccxt public API but has never been run. Symbol formats, ticker field names
  and rate-limit behaviour vary by exchange.
- **Real Telegram delivery.** The transport that makes the actual HTTP call has
  never sent a message. Everything around it is tested.
- **A live smoke test.** No cycle has ever run against real prices.
- **The broad market-cap universe provider.**
  `crypto_edge/data/broad_universe.py::CoinGeckoUniverseProvider` is written
  against the public CoinGecko markets endpoint but has never been executed.
  Until it succeeds once, `require_broad_universe = true` means **no new
  entries will be opened** — that is the intended fail-closed behaviour, not a
  bug. `selfcheck` reports it explicitly.
- **Real ticker-timestamp behaviour.** `max_quote_age_s = 90` is a defensible
  default, not a measurement. `verify-live` samples real tickers and tells you
  the median and maximum observed age on your venue, and whether that venue
  stamps its tickers at all. Do not raise the limit to make something pass —
  read the finding first.
- **Real ticker collisions.** The collision machinery is tested offline against
  a constructed clash. Which tickers actually collide in CoinGecko's top ~1000
  is a live fact; `verify-live` lists them so you can pin any you want traded.

Run `python -m crypto_edge.cli selfcheck` first — it checks exactly these three
things, and will tell you which one is broken if any are.

## Layout

```
crypto_edge/
  engine.py          cycle orchestration
  selfcheck.py       startup gate
  cli.py             commands
  backtest.py        same strategy object, bar-sliced history
  data/              feed protocol, ccxt feed, fixture feed, broad (market-cap)
                     asset universe, venue universe builder
  verify_live.py     live-network verification harness (cli verify-live)
  strategy/          regime, scoring, trend_breakout
  execution/         paper broker: sizing, fills, fees, slippage, stops
  portfolio/         account, risk manager, stop logic
  storage/           schema and repository
  intel/             news, token events, derivatives
  research/          decision journal, counterfactuals
  notify/            formatters, Telegram notifier
config/config.toml   all parameters, commented
scripts/             start/stop/status wrappers, offline smoke test
tests/               685 tests
```

## Safety notes

- `load_config()` forces `mode = "PAPER"` and `live_trading_enabled = False`
  regardless of what the TOML says. Editing those values does nothing.
- No exchange API keys are read, stored, or constructed anywhere.
- `.env` is gitignored. Do not commit it.
- The bot fails closed: missing BTC reference data, stale candles, corrupt
  series, an unreachable feed, a quote it cannot trust, or no valid broad asset
  universe all suspend NEW ENTRIES rather than guessing. Open positions
  continue to be managed in every one of those cases — losing a data source
  must never strand risk already on the books.
