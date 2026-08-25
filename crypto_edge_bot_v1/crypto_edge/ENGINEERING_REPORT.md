# ENGINEERING REPORT — crypto_edge_bot_v1

Build date: 2026-08-22
Environment: Python 3.12.3, numpy 2.4.4, SQLite 3.45.1, **no outbound network**

---

## WHAT YOU CHANGED

Built from scratch. Nothing from the previous bots was reused, per your
instruction. 38 modules, ~7,900 lines including tests.

**Foundation.** UTC millisecond epochs everywhere, with a deterministic
`candle_id(symbol, timeframe, open_ms)` that is the backbone of duplicate
protection. All indicators are pure numpy, causal by construction, with
explicit NaN warm-up rather than silently truncated windows.

**Persistence.** SQLite in WAL mode with `synchronous=FULL`. Sixteen tables.
Writes go through a `BEGIN IMMEDIATE` transaction context manager. Position
opens and closes are atomic, and the candle claim happens *inside* the same
transaction as the position insert — so a crash between the two is impossible
rather than merely unlikely.

**Execution accounting.** The convention is explicit and tested end to end:

```
gross     = (exit_ref  − entry_ref)  × qty
slippage  = (entry_fill − entry_ref) × qty + (exit_ref − exit_fill) × qty
net       = gross − slippage − fees   ≡  (exit_fill − entry_fill) × qty − fees
```

Both forms are asserted equal in the tests. Sizing applies the risk budget
first, then caps by max position percent, remaining exposure room, and
affordable cash *including the entry fee*, then rounds **down** to exchange
precision and checks minimum notional.

**Strategy.** A trend-breakout system with hard filters applied in cost order
before any scoring happens, then a nine-component weighted score. Each weight
carries a written justification in `WEIGHTS`/`JUSTIFICATIONS` — those are
documented opinions, not tuned parameters, and no optimisation has been run.
Strategies are pure functions returning a `Signal`; the portfolio manager alone
decides what is bought.

**Look-ahead architecture.** Your `event_time` + `observed_at` design is kept
throughout, on news, token events, funding, open interest and liquidations.
Every visibility query filters on `observed_at`.

**Research.** Every evaluation is journaled with its full feature snapshot and
rejection reason. Rejected signals are tracked as counterfactuals at six
horizons, stored separately, flagged `hypothetical=1`, structurally unable to
create a position.

**Interfaces.** CLI with nine commands, structured JSON logging across six
channels, Telegram with an injectable transport, a startup self-check gate, and
a backtester that calls the *same strategy object* the live engine calls.

---

## BUGS FOUND

**1. Self-check logged non-critical failures at ERROR.** A failed
WARNING-severity check (e.g. Telegram unreachable) logged at the same level as
database corruption. Anyone watching these logs would learn to ignore both.
Fixed: log level now tracks check severity.

**2. Self-check log messages read as success claims.** The message was
`selfcheck: <check name>`, so a failing configuration check emitted
`ERROR selfcheck: configuration valid` — which parses as an assertion that the
configuration *is* valid. Fixed to `selfcheck FAILED: configuration valid`.

**3. Two engine tests were silently vacuous.** They appended a crash bar to a
fixture series without shifting the series start, leaving that bar in the
future. `drop_unclosed` correctly discarded it, so the stop was never breached
and the tests were asserting against a scenario that never occurred. Fixed with
a `rebuild_feed` helper. Worth noting the *bot* was right here — this was the
look-ahead guard working correctly and the test being wrong.

**4. A test asserted a wrong accounting identity.** It reconciled cash against
`net_pnl + qty × entry_fill`, omitting the entry fee. Replaced with the
stronger and simpler round-trip check: with the book flat,
`cash − starting_equity ≡ net_pnl`, exactly.

**5. Synthetic fixtures were unrealistically smooth.** A perfectly monotonic
`cumsum` uptrend pins RSI and ADX at 100, which no real market produces and
which the exhaustion filter correctly rejected. Fixtures now use noisy trends
with genuine pullbacks. This one is a caution about all synthetic testing: it
is easy to build data that flatters the strategy, and easy to build data no
strategy could trade.

Two things I reported to you mid-build and was **wrong** about, corrected here
for the record: I claimed `selfcheck` exited 0 despite failing, and that it
logged a *passing* check at ERROR. The first was my measurement error — I read
`$?` after a pipeline, so I was reading `tail`'s exit code. It exits 1
correctly, and `start.sh`'s guard does work. The second was a misreading of the
message text (defect 2 above), not a wrong severity.

---

## TEST RESULTS

**431 tests, all passing.** Run: `python -m crypto_edge.cli test`

| Module | Tests | Covers |
|---|---:|---|
| `test_execution.py` | 21 | Sizing, caps, fees, slippage direction, stop fills, gap-through, P&L identity |
| `test_account.py` | 17 | Cash reconciliation, atomicity, duplicate guards, full restart persistence |
| `test_risk_and_stops.py` | 24 | Circuit breakers, entry gates, correlation, stop ratcheting, time stops |
| `test_lookahead.py` | 25 | Candle closure, indicator causality, determinism, backtest slicing |
| `test_research_intel.py` | 45 | `observed_at` visibility, classification, blocking, counterfactuals, venue universe filters |
| `test_performance.py` | 22 | Win rate, profit factor, expectancy, drawdown, sample-size honesty |
| `test_telegram.py` | 17 | Formatter content, retry, fail-safe, error cooldown |
| `test_telegram_outbox.py` | 23 | PENDING→SENT outbox, total outage, recovery, restart mid-failure, duplicate suppression, v1→v2 migration |
| `test_entry_quote_gate.py` | 28 | Missing/invalid/stale/malformed/crossed/wide quotes fail closed; exits unaffected |
| `test_entry_sizing.py` | 24 | Expected fill == real fill, sizing on the fill, revalidation, slippage/rounding/allocation/fee interaction |
| `test_broad_universe.py` | 36 | Market-cap universe, intersection, caching + provenance, outage fallback, fail-closed entries |
| `test_engine.py` | 12 | End-to-end entry, stop exit, stale data, breakers, restart mid-position |
| `test_selfcheck.py` | 12 | Fail-closed behaviour, critical vs warning, log severity |
| `test_live_adapter.py` | 45 | CCXT precision modes and tick rounding, quote-timestamp provenance, OHLCV parsing |
| `test_symbol_collisions.py` | 26 | Ticker collisions, identity-based resolution, overrides, cached ambiguity |
| `test_verify_live.py` | 26 | The live-verification harness itself, including credential masking |
| `test_windows_operability.py` | 20 | UTF-8 BOM in `.env`, redirected output encoding, exchange override |
| `test_repository_integrity.py` | 8 | Every module is committed and importable; no source file is gitignored |

The tests worth singling out, because they are the ones that would catch a real
loss of money:

- Round-trip cash change equals net P&L to 6 decimal places, for wins and losses
- A gap through the stop fills at the candle **open**, not the stop price
- The same candle cannot produce two entries, and that guard survives a restart
- Future bars do not change a past decision (determinism under replay)
- The Donchian channel excludes the current bar
- An unobserved news event is invisible, even if it already happened
- A counterfactual can never create a position

`scripts/smoke_test.py` runs the full lifecycle offline. Latest output:

```
ENTERED SOL/USDT qty=10.4654 @ 143.443 stop=139.254 risk=$42.64
restored 1 position(s) from SQLite
CLOSED SOL/USDT reason=stop_loss
  gross $-42.64  fees $2.22  slippage $3.39  NET $-48.24
  identity check: gross - fees - slippage = net  OK
equity reconciliation drift: $0.0000000000
```

The gross loss landing on exactly the $42.64 risk budget is the sizing maths
hitting its target. The extra $5.61 is friction — which is precisely what most
backtests omit and why paper results usually look better than they are.

---

## CURRENT STRATEGY

`trend_breakout` v1.0.0, long-only, on 1h entries with a 4h regime filter.

Hard filters, in order: sufficient history → data sanity → indicator readiness
→ not on the no-trade list → BTC regime not bear/unknown → HTF regime bullish →
EMA alignment (fast > slow > trend) → price above fast EMA → Donchian(48)
breakout excluding the current bar → ADX ≥ 20 → RSI ≤ 78 → relative volume ≥
1.2 → extension ≤ 3.5 ATR → score ≥ 55.

Score components and weights: HTF trend .18, breakout quality .15, ADX .12,
momentum .12, relative strength .12, relative volume .10, extension .08,
liquidity .08, market context .05.

Exits: chandelier trail at 3.0 ATR (ratchet-only, never widens), breakeven at
1R plus 0.1R offset, trend invalidation, time stop at 96 bars if under 0.5R.

**These parameters are reasoned defaults, not optimised ones.** No walk-forward
or parameter search has been run against real data. `backtest.py` provides
`walk_forward_windows()` and `parameter_stress()` for when you have history —
the latter checks whether the neighbourhood around each parameter is also
profitable, which is how you distinguish a robust setting from a cliff edge.

---

## PAPER ACCOUNT CONFIGURATION

| | |
|---|---|
| Starting equity | $10,000 |
| Risk per trade | 0.5% of equity at the initial stop ($50 at start) |
| Max position | 15% of equity |
| Max portfolio exposure | 60% |
| Max open positions | 6 |
| Max new entries per cycle | 2 |
| Daily loss limit | 3% — halts new entries, clears at UTC midnight |
| Max drawdown | 20% — kill switch, does **not** auto-reset |
| Max correlation | 0.85 against the worst open position |
| Taker fee | 7.5 bps |
| Slippage | 6 bps normal, 15 bps on stops |
| Universe | Top 200 by volume → filtered → max 40 tradable |

All editable in `config/config.toml`. `mode` and `live_trading_enabled` are
**forced** to `PAPER`/`false` in `load_config()` regardless of the file
contents — editing them there does nothing, deliberately.

---

## TELEGRAM STATUS

Implemented and tested up to, but not including, the network call.

Message types: bot start, entry, stop update, exit with full cost breakdown,
circuit breaker, heartbeat, error, daily report. Dedupe keys are claimed in
SQLite *before* sending, so a race between two code paths cannot double-send,
and the suppression survives a restart. Exponential backoff on retry. Errors
are rate-limited with a 15-minute cooldown per distinct error, so one failing
loop cannot flood the chat. **The notifier never raises** — a Telegram outage
degrades notifications, not trading.

**Delivery itself is unverified.** `RequestsTransport` has never made a real
HTTP request. Add credentials to `.env` and run `selfcheck`; it tests
connectivity and reports as a WARNING, not a failure, since losing Telegram
should never stop the trader.

---

## HOW TO START IT

```bash
cd crypto_edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

python -m crypto_edge.cli selfcheck    # verify first
python -m crypto_edge.cli start        # then trade
```

Background: `scripts/start.sh` (runs the self-check, refuses to start if it
fails, writes a pid file).

## HOW TO STOP IT

Ctrl-C in the foreground, or `scripts/stop.sh` for a detached process. Both
send a graceful signal: the engine finishes its current cycle, commits, and
exits. State is committed every cycle, so even a hard kill loses at most the
cycle in flight.

## HOW TO CHECK PERFORMANCE

```bash
python -m crypto_edge.cli status                    # quick snapshot
python -m crypto_edge.cli positions                 # open positions and stops
python -m crypto_edge.cli performance               # full report
python -m crypto_edge.cli performance --categories  # breakdowns by bucket
python -m crypto_edge.cli research                  # decisions and counterfactuals
python -m crypto_edge.cli export --out trades.csv   # raw ledger
```

Ratios (Sharpe, Sortino, Calmar) carry an explicit `reliable` flag and are
labelled `SAMPLE TOO SMALL` below 30 trades or 20 trading days. Category
breakdowns carry `sufficient_sample` flags below 10. This is deliberate: the
most likely way this bot misleads you is by showing a confident-looking Sharpe
ratio computed from eleven trades.

---

## REMAINING RISKS

**1. Live data is entirely unverified.** This is the big one. `ccxt_feed.py`
has never executed. Symbol formats, ticker field names, pagination and
rate-limit behaviour vary between exchanges, and Binance may geo-block you.
Expect to debug this. Run `selfcheck` before anything else.

**2. Paper fills are a model, not reality.** Fills are simulated against candle
data with a fixed slippage assumption. Real fills depend on book depth at the
moment of execution. Thin markets will be worse than modelled; the 15 bps stop
slippage is a guess, and in a genuine liquidation cascade it will be optimistic.

**3. No real-data validation of the strategy.** The strategy is untested
against actual price history. It may simply not be profitable. Paper trading
is how you find out — and you need dozens of trades before the numbers mean
anything.

**4. Long-only, trend-following, correlated.** In a broad crypto drawdown, the
BTC regime filter should keep it flat, but if it is caught holding, six
positions in altcoins is close to one leveraged bet on beta. The 0.85
correlation cap helps and does not eliminate this.

**5. Single-threaded and sequential by design.** A slow exchange response
delays the whole cycle. This is a deliberate trade of throughput for the
ability to reason about ordering, but it means latency shows up as missed bars
rather than as queued work.

**6. News and derivatives are architecture, not data.** Both ship disabled
because no provider is wired. The storage, look-ahead guard, classification and
blocking logic are built and tested; the feeds are not. Stated precisely, since
this is easy to overstate: no provider is constructed anywhere in the running
system (`NewsEngine(repo, [])`, `DerivativesEngine(repo, [])`), and
`NewsEngine.poll()` / `DerivativesEngine.poll()` are never called by the engine
cycle, so `intel.news_poll_minutes` and `intel.derivatives_poll_minutes`
currently drive nothing. The read paths (`context_for`, `blocking_event`) are
live but query tables no production code writes to. **A news event cannot
currently stop a trade, and no funding or open-interest data is being
recorded.** See the status table in the README.

**7. Synthetic test data is a limited proxy.** 431 passing tests prove the
system is internally consistent and behaves correctly against data I generated.
They cannot prove it behaves correctly against data reality generates.

---

## LIVE-READINESS PASS (adapter correctness)

Two defects were found by auditing the CCXT adapter against the installed
library rather than by reading it:

**A. Precision was wrong on every major venue.** CCXT reports market
granularity in three modes. `TICK_SIZE` — used by binance, kraken, coinbase,
kucoin, okx, bybit and bitstamp in current CCXT — expresses it as an absolute
tick *float* (`0.00001`). The adapter tested `isinstance(raw, int)`, which is
False for a float, and fell back to 8 decimal places. Quantities were therefore
sized three decimal places finer than the venue permits, and ticks that are not
powers of ten (`0.05`, `0.5`) could not be represented at all. `MarketMeta` now
carries real tick sizes, the broker rounds to them (down for quantity, nearest
for price), and all three precision modes are handled.

**B. A missing ticker timestamp was silently replaced with local time.** CCXT
normalises an absent venue timestamp to `None`; substituting `now_ms()` without
recording that made a quote of unknown vintage read as 0 seconds old to the
entry staleness check — laundering an unknown into a pass. Quotes now carry
`ts_source`, and the entry gate *skips* the age check on a locally-stamped
quote (recording `age_verified = False`) rather than scoring it, leaving the
clock-independent price-deviation check to guard those venues.

**C. Ticker symbols are not globally unique.** The universe mapped market-cap
assets to exchange markets by ticker alone, and de-duplicated by ticker —
silently keeping whichever ranked higher and destroying the evidence that the
ticker was ambiguous. Assets now carry the provider's stable id, the collision
scan reaches to rank ~1000 while trading stops at 200 (so a clash below the
cut-off is still visible), an ambiguous ticker is refused rather than guessed,
and `universe.broad_symbol_overrides` pins one deterministically. The ambiguity
map is cached with the ranking, so an outage fallback is as careful as a live
fetch.

---

## OPERABILITY PASS (Windows)

Three issues that would have bitten a non-developer following the setup steps:

**A. A Notepad-saved `.env` silently disabled Telegram.** Windows Notepad
writes a UTF-8 byte order mark. Read as plain UTF-8 the BOM becomes a `\ufeff`
character glued to the first key, so `TELEGRAM_BOT_TOKEN` was stored as
`\ufeffTELEGRAM_BOT_TOKEN`, never found, and the bot ran with notifications
quietly off — no error anywhere. Reproduced, then fixed by reading `.env` and
`config.toml` as `utf-8-sig`.

**B. Non-ASCII output crashed when redirected.** Python writes to a Windows
console through the Unicode API but falls back to the system code page when
output is piped or redirected — so a single emoji in a status line raised
`UnicodeEncodeError` exactly when someone was capturing output to send to
support. The CLI entrypoint now reconfigures stdout/stderr to UTF-8 with a
replacement error handler.

**C. Changing exchange required editing TOML.** Venue switching is the most
common operational change (geo-blocks, outages), and a mistyped TOML line is a
worse failure than a wrong venue. Added `--exchange` / `--quote` flags and the
`CRYPTO_EDGE_EXCHANGE` / `CRYPTO_EDGE_QUOTE` environment overrides.

---

## NEXT RECOMMENDED IMPROVEMENT

**Run it on live data for two weeks and change nothing.**

Not a feature. The single highest-value next step is accumulating a real
decision log — including the rejections and their counterfactuals — because
every subsequent decision depends on evidence this system does not yet have.
Specifically, in order:

1. `selfcheck` against your exchange, fix whatever ccxt surfaces.
2. Run with Telegram on. Watch for message volume; tune
   `stop_update_min_pct` if it is chatty.
3. After ~30 closed trades, read `performance --categories` **and**
   `research`. The counterfactual report is the most valuable output here: if
   one rejection reason consistently shows positive hypothetical returns, a
   filter is costing you money. That is a data-driven parameter change rather
   than a guess.
4. Only then consider tuning, and use `parameter_stress()` to check the
   neighbourhood rather than the point.

The temptation will be to adjust parameters after a few losing trades. With a
0.5% risk budget, a run of six losses is entirely ordinary and tells you
nothing. Resisting that is worth more than any code I could add.

---

## VERIFICATION SUMMARY

### VERIFIED OFFLINE
Sizing, fees, slippage, P&L identity, stop fills including gaps; account
persistence and restart recovery; risk gates and circuit breakers; stop
ratcheting and time stops; look-ahead protection and indicator causality;
determinism under replay; universe filtering; news/event visibility and
blocking; research journal and counterfactual isolation; performance metrics
and sample-size flags; Telegram formatting, retry, dedupe and fail-safe logic;
the self-check gate; full end-to-end lifecycle on synthetic data.

### REQUIRES VERIFICATION ON YOUR MACHINE
Live market data fetching via ccxt (never executed — the library could not even
be installed here). Real Telegram message delivery (the HTTP transport has
never made a request). A live smoke test (no cycle has run against real
prices).

I did not test these and am not claiming they work. The self-check exists
specifically to tell you which of them is broken.
