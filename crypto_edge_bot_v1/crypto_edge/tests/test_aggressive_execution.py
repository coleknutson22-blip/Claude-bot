"""Strategy B execution: shorts that pay, exits that cannot fail open, restart.

TWO THINGS THIS FILE GUARDS THAT THE SOURCE BOT GOT WRONG
---------------------------------------------------------
1. ITS EXITS COULD FAIL OPEN. The time-based exit was delegated to a local LLM;
   when the model was unreachable the call returned None, no exit fired, and the
   position was held indefinitely with no error. Every exit here is arithmetic
   on a price and a clock, and a test drives each one.

2. IT HAD NO LIQUIDATION MODEL. `position_pnl` was unbounded and equity simply
   absorbed it. A long cannot lose more than it paid -- price stops at zero -- but
   a SHORT has no such floor: at 1x its loss exceeds its collateral once price
   merely doubles. The forced close is what stands in for the margin call a
   paper ledger does not have.

Financing is charged because a free short is not a short. Kraken spot cannot be
sold short at all; a real one needs the margin product and pays for it, and
charging nothing would flatter every short in the long-versus-short comparison
this strategy exists to make.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import AggressiveCfg
from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import Candle, MarketMeta, Position
from crypto_edge.portfolio import aggressive_exits as ex
from crypto_edge.portfolio.account import PaperAccount
from helpers import open_repo, temp_repo

B = "aggressive_momentum_v2"
META = MarketMeta("X/USD", "X", "USD", True, 6, 6, 1e-6, 5.0)
HOUR = 3_600_000
DAY = 24 * HOUR
CFG = AggressiveCfg()


def position(side="long", entry=100.0, qty=10.0, stop=None, t0=0, **over):
    d = 1 if side == "long" else -1
    stop = stop if stop is not None else entry * (1 - 0.02 * d)
    kw = dict(id="p1", symbol="X/USD", strategy=B, strategy_version="1",
              side=side, qty=qty, entry_ref_price=entry, entry_fill_price=entry,
              entry_ms=t0, entry_fee=0.0, entry_slippage=0.0,
              initial_stop=stop, current_stop=stop, highest_price=entry,
              lowest_price=entry, risk_amount=0.0, candle_id="c1", journal={})
    kw.update(over)
    return Position(**kw)


def cfg(**over):
    c = AggressiveCfg()
    for k, v in over.items():
        setattr(c, k, v)
    return c


class TestRIsSignedByDirection(unittest.TestCase):
    def test_r_is_positive_on_both_sides(self):
        self.assertAlmostEqual(ex.risk_per_unit(position("long")), 2.0)
        self.assertAlmostEqual(ex.risk_per_unit(position("short")), 2.0)

    def test_progress_is_measured_in_the_trades_favour(self):
        self.assertAlmostEqual(ex.progress_r(position("long"), 104.0), 2.0)
        self.assertAlmostEqual(ex.progress_r(position("short"), 96.0), 2.0)

    def test_an_adverse_move_is_negative_on_both_sides(self):
        self.assertLess(ex.progress_r(position("long"), 98.0), 0)
        self.assertLess(ex.progress_r(position("short"), 102.0), 0)

    def test_the_target_sits_on_the_profitable_side(self):
        self.assertAlmostEqual(ex.target_price(position("long"), 2.0), 104.0)
        self.assertAlmostEqual(ex.target_price(position("short"), 2.0), 96.0)


class TestStopsOnlyRatchet(unittest.TestCase):
    def test_breakeven_moves_the_stop_to_entry_plus_an_offset(self):
        p = position("long")
        upd = ex.update_stop(p, 102.0, atr=1.0, breakeven_at_r=1.0,
                             breakeven_offset_r=0.1, trail_start_r=99.0,
                             trail_atr_mult=2.5)
        self.assertTrue(upd.changed)
        self.assertEqual(upd.kind, "breakeven")
        self.assertAlmostEqual(upd.new_stop, 100.2)

    def test_breakeven_is_mirrored_for_a_short(self):
        p = position("short")
        upd = ex.update_stop(p, 98.0, atr=1.0, breakeven_at_r=1.0,
                             breakeven_offset_r=0.1, trail_start_r=99.0,
                             trail_atr_mult=2.5)
        self.assertTrue(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 99.8)

    def test_the_trail_only_starts_once_genuinely_in_profit(self):
        p = position("long")
        quiet = ex.update_stop(p, 100.5, atr=1.0, breakeven_at_r=99.0,
                               breakeven_offset_r=0.1, trail_start_r=1.5,
                               trail_atr_mult=2.0)
        self.assertFalse(quiet.changed, "0.25R is not enough to start trailing")
        running = ex.update_stop(p, 110.0, atr=1.0, breakeven_at_r=99.0,
                                 breakeven_offset_r=0.1, trail_start_r=1.5,
                                 trail_atr_mult=2.0)
        self.assertTrue(running.changed)
        self.assertEqual(running.kind, "chandelier")
        self.assertAlmostEqual(running.new_stop, 108.0)

    def test_the_short_trail_is_the_mirror(self):
        p = position("short", lowest_price=90.0)
        upd = ex.update_stop(p, 90.0, atr=1.0, breakeven_at_r=99.0,
                             breakeven_offset_r=0.1, trail_start_r=1.5,
                             trail_atr_mult=2.0)
        self.assertTrue(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 92.0)

    def test_a_stop_never_loosens(self):
        p = position("long", current_stop=105.0, highest_price=110.0)
        upd = ex.update_stop(p, 101.0, atr=1.0, breakeven_at_r=1.0,
                             breakeven_offset_r=0.1, trail_start_r=1.5,
                             trail_atr_mult=2.0)
        self.assertFalse(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 105.0)

    def test_a_short_stop_never_loosens_upward(self):
        p = position("short", current_stop=95.0, lowest_price=90.0)
        upd = ex.update_stop(p, 99.0, atr=1.0, breakeven_at_r=1.0,
                             breakeven_offset_r=0.1, trail_start_r=1.5,
                             trail_atr_mult=2.0)
        self.assertFalse(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 95.0)

    def test_is_better_stop_is_direction_aware(self):
        self.assertTrue(ex.is_better_stop(99.0, 98.0, 1))
        self.assertFalse(ex.is_better_stop(97.0, 98.0, 1))
        self.assertTrue(ex.is_better_stop(101.0, 102.0, -1))
        self.assertFalse(ex.is_better_stop(103.0, 102.0, -1))


class TestTheDeterministicExitSet(unittest.TestCase):
    def test_the_profit_target_fires_at_its_r_multiple(self):
        self.assertEqual(ex.check_exit(position("long"), 104.0, now_ms=0,
                                       cfg=cfg()), ex.TARGET)
        self.assertIsNone(ex.check_exit(position("long"), 103.0, now_ms=0,
                                        cfg=cfg()))

    def test_the_target_is_mirrored_for_a_short(self):
        self.assertEqual(ex.check_exit(position("short"), 96.0, now_ms=0,
                                       cfg=cfg()), ex.TARGET)

    def test_momentum_invalidation_fires_when_structure_flips(self):
        r = ex.check_exit(position("long"), 100.0, now_ms=0, cfg=cfg(),
                          ema_struct=-1.0)
        self.assertEqual(r, ex.MOMENTUM)

    def test_momentum_invalidation_is_mirrored(self):
        r = ex.check_exit(position("short"), 100.0, now_ms=0, cfg=cfg(),
                          ema_struct=1.0)
        self.assertEqual(r, ex.MOMENTUM)

    def test_favourable_structure_does_not_trigger_an_exit(self):
        self.assertIsNone(ex.check_exit(position("long"), 100.0, now_ms=0,
                                        cfg=cfg(), ema_struct=1.0))

    def test_a_hostile_regime_closes_a_long(self):
        r = ex.check_exit(position("long"), 100.0, now_ms=0, cfg=cfg(),
                          btc_regime="bear")
        self.assertEqual(r, ex.REGIME)

    def test_a_hostile_regime_closes_a_short(self):
        r = ex.check_exit(position("short"), 100.0, now_ms=0, cfg=cfg(),
                          btc_regime="bull")
        self.assertEqual(r, ex.REGIME)

    def test_a_favourable_regime_does_not(self):
        self.assertIsNone(ex.check_exit(position("long"), 100.0, now_ms=0,
                                        cfg=cfg(), btc_regime="bull"))

    def test_the_regime_exit_is_switchable(self):
        self.assertIsNone(ex.check_exit(
            position("long"), 100.0, now_ms=0,
            cfg=cfg(exit_on_hostile_regime=False), btc_regime="bear"))

    def test_the_time_stop_fires_at_its_horizon(self):
        r = ex.check_exit(position("long"), 101.0, now_ms=int(8.1 * HOUR),
                          cfg=cfg())
        self.assertEqual(r, ex.TIME)

    def test_a_trade_going_nowhere_is_cut_earlier(self):
        r = ex.check_exit(position("long"), 100.2, now_ms=int(4.5 * HOUR),
                          cfg=cfg())
        self.assertEqual(r, ex.TIME_EARLY,
                         "0.1R after four hours is capital doing nothing")

    def test_a_trade_making_progress_is_left_alone_at_four_hours(self):
        self.assertIsNone(ex.check_exit(position("long"), 102.0,
                                        now_ms=int(4.5 * HOUR), cfg=cfg()))

    def test_the_horizon_is_hours_not_days(self):
        self.assertLessEqual(CFG.time_stop_hours, 24)
        self.assertLess(CFG.time_stop_early_hours, CFG.time_stop_hours)

    def test_every_exit_reason_is_reachable_without_a_model(self):
        """The defect in the source bot: an exit that needed a network call."""
        seen = {
            ex.check_exit(position("long"), 104.0, now_ms=0, cfg=cfg()),
            ex.check_exit(position("long"), 100.0, now_ms=0, cfg=cfg(),
                          ema_struct=-1.0),
            ex.check_exit(position("long"), 100.0, now_ms=0, cfg=cfg(),
                          btc_regime="bear"),
            ex.check_exit(position("long"), 101.0, now_ms=int(9 * HOUR), cfg=cfg()),
            ex.check_exit(position("long"), 100.1, now_ms=int(5 * HOUR), cfg=cfg()),
            ex.check_exit(position("short", entry=100.0), 190.0, now_ms=0, cfg=cfg()),
        }
        self.assertEqual(
            seen, {ex.TARGET, ex.MOMENTUM, ex.REGIME, ex.TIME, ex.TIME_EARLY,
                   ex.FORCED_SHORT})

    def test_a_quiet_position_is_not_exited(self):
        self.assertIsNone(ex.check_exit(position("long"), 100.5, now_ms=HOUR,
                                        cfg=cfg(), ema_struct=0.5,
                                        btc_regime="neutral"))


class TestShortFinancing(unittest.TestCase):
    def test_a_long_never_pays_borrow(self):
        for hours in (0, 1, 24, 240):
            self.assertEqual(
                ex.financing_cost(position("long"), hours * HOUR, 15.0), 0.0)

    def test_a_short_accrues_pro_rata(self):
        p = position("short")                     # $1,000 notional
        self.assertAlmostEqual(ex.financing_cost(p, DAY, 15.0), 1.50, places=6)
        self.assertAlmostEqual(ex.financing_cost(p, DAY // 2, 15.0), 0.75, places=6)
        self.assertAlmostEqual(ex.financing_cost(p, 0, 15.0), 0.0)

    def test_a_zero_rate_is_free_but_only_when_configured(self):
        self.assertEqual(ex.financing_cost(position("short"), DAY, 0.0), 0.0)
        self.assertGreater(CFG.short_borrow_bps_per_day, 0.0,
                           "the shipped default must not give shorts free money")

    def test_financing_scales_with_notional(self):
        small = ex.financing_cost(position("short", qty=1.0), DAY, 15.0)
        big = ex.financing_cost(position("short", qty=10.0), DAY, 15.0)
        self.assertAlmostEqual(big, small * 10.0, places=9)

    def test_financing_reduces_open_equity_not_just_the_close(self):
        """Charging only at exit would let a short read high until it was too
        late to act on the cost."""
        repo, _ = temp_repo()
        broker = PaperBroker(7.5, 6.0, 15.0)
        acct = PaperAccount(repo, broker, 10_000.0, B, borrow_bps_per_day=15.0)
        fill = broker.entry_fill("X/USD", 10.0, 100.0, -1, meta=META)
        pos = acct.open_position(
            symbol="X/USD", strategy=B, strategy_version="1", qty=10.0,
            ref_price=100.0, fill=fill, initial_stop=102.0, risk_amount=20.0,
            candle_id="c1", signal_score=70.0, journal={}, side="short")
        self.assertIsNotNone(pos)
        self.assertGreater(acct.financing(pos, pos.entry_ms + DAY), 0.0)

    def test_a_short_round_trip_reconciles_including_borrow(self):
        repo, _ = temp_repo()
        broker = PaperBroker(7.5, 6.0, 15.0)
        acct = PaperAccount(repo, broker, 10_000.0, B, borrow_bps_per_day=15.0)
        start = acct.cash()
        fill = broker.entry_fill("X/USD", 10.0, 100.0, -1, meta=META)
        pos = acct.open_position(
            symbol="X/USD", strategy=B, strategy_version="1", qty=10.0,
            ref_price=100.0, fill=fill, initial_stop=102.0, risk_amount=20.0,
            candle_id="c1", signal_score=70.0, journal={}, side="short")
        exit_fill = broker.exit_fill("X/USD", 10.0, 95.0, -1, meta=META,
                                     ts_ms=pos.entry_ms + DAY)
        trade = acct.close_position(pos, exit_fill, "target")
        self.assertGreater(trade.financing, 0.0, "the short paid to borrow")
        self.assertAlmostEqual(acct.cash() - start, trade.net_pnl, places=6)

    def test_the_trade_ledger_records_financing_separately(self):
        repo, _ = temp_repo()
        broker = PaperBroker(7.5, 6.0, 15.0)
        acct = PaperAccount(repo, broker, 10_000.0, B, borrow_bps_per_day=15.0)
        fill = broker.entry_fill("X/USD", 10.0, 100.0, -1, meta=META)
        pos = acct.open_position(
            symbol="X/USD", strategy=B, strategy_version="1", qty=10.0,
            ref_price=100.0, fill=fill, initial_stop=102.0, risk_amount=20.0,
            candle_id="c1", signal_score=70.0, journal={}, side="short")
        acct.close_position(pos, broker.exit_fill("X/USD", 10.0, 99.0, -1,
                                                  meta=META,
                                                  ts_ms=pos.entry_ms + DAY),
                            "target")
        row = repo.get_trades(B)[0]
        self.assertGreater(row["financing"], 0.0)
        self.assertEqual(row["side"], "short")

    def test_a_long_trade_records_zero_financing(self):
        repo, _ = temp_repo()
        broker = PaperBroker(7.5, 6.0, 15.0)
        acct = PaperAccount(repo, broker, 10_000.0, B, borrow_bps_per_day=15.0)
        fill = broker.entry_fill("X/USD", 10.0, 100.0, 1, meta=META)
        pos = acct.open_position(
            symbol="X/USD", strategy=B, strategy_version="1", qty=10.0,
            ref_price=100.0, fill=fill, initial_stop=98.0, risk_amount=20.0,
            candle_id="c1", signal_score=70.0, journal={}, side="long")
        acct.close_position(pos, broker.exit_fill("X/USD", 10.0, 105.0, 1,
                                                  meta=META,
                                                  ts_ms=pos.entry_ms + DAY),
                            "target")
        self.assertEqual(repo.get_trades(B)[0]["financing"], 0.0)


class TestForcedShortClose(unittest.TestCase):
    """A short's loss is unbounded. Nothing in a paper ledger stops it."""

    def test_the_loss_fraction_tracks_the_collateral(self):
        p = position("short", entry=100.0, qty=10.0)   # $1,000 collateral
        for price, want in ((100.0, 0.0), (110.0, 0.10), (150.0, 0.50),
                            (180.0, 0.80), (200.0, 1.00)):
            self.assertAlmostEqual(ex.short_loss_fraction(p, price), want,
                                   places=6, msg=f"price {price}")

    def test_a_long_never_reports_a_collateral_breach(self):
        self.assertEqual(ex.short_loss_fraction(position("long"), 1.0), 0.0)
        self.assertFalse(ex.force_close_short(position("long"), 1.0,
                                              at_loss_pct=80.0))

    def test_the_close_fires_before_the_collateral_is_gone(self):
        p = position("short", entry=100.0, qty=10.0)
        self.assertFalse(ex.force_close_short(p, 175.0, at_loss_pct=80.0))
        self.assertTrue(ex.force_close_short(p, 181.0, at_loss_pct=80.0))
        self.assertLess(181.0, 200.0,
                        "and does so with collateral still covering the loss")

    def test_financing_counts_toward_the_breach(self):
        p = position("short", entry=100.0, qty=10.0)
        without = ex.short_loss_fraction(p, 179.0, 0, 0.0)
        with_borrow = ex.short_loss_fraction(p, 179.0, 30 * DAY, 15.0)
        self.assertGreater(with_borrow, without)

    def test_the_forced_close_outranks_every_other_exit(self):
        """Solvency is decided before the trade thesis is even consulted."""
        p = position("short", entry=100.0, qty=10.0)
        r = ex.check_exit(p, 190.0, now_ms=int(99 * HOUR), cfg=cfg(),
                          ema_struct=-1.0, btc_regime="bull")
        self.assertEqual(r, ex.FORCED_SHORT)

    def test_the_threshold_is_configurable_and_below_one_hundred(self):
        self.assertLess(CFG.short_force_close_at_loss_pct, 100.0)
        p = position("short", entry=100.0, qty=10.0)
        self.assertTrue(ex.force_close_short(p, 151.0, at_loss_pct=50.0))


class TestStopResolutionAgainstTheCandle(unittest.TestCase):
    def test_a_long_stop_fills_at_the_open_when_the_market_gaps(self):
        broker = PaperBroker(7.5, 6.0, 15.0)
        candle = Candle(0, 95.0, 96.0, 90.0, 94.0, 100.0)
        fill = broker.stop_exit("X/USD", 10.0, 98.0, candle, META, direction=1)
        self.assertIsNotNone(fill)
        self.assertEqual(fill.reason, "stop_gap")
        self.assertLess(fill.fill_price, 98.0)

    def test_a_short_stop_gaps_upward(self):
        broker = PaperBroker(7.5, 6.0, 15.0)
        candle = Candle(0, 105.0, 110.0, 104.0, 108.0, 100.0)
        fill = broker.stop_exit("X/USD", 10.0, 102.0, candle, META, direction=-1)
        self.assertIsNotNone(fill)
        self.assertEqual(fill.reason, "stop_gap")
        self.assertGreater(fill.fill_price, 102.0,
                           "a short covering a gap pays MORE, not less")

    def test_an_untouched_stop_does_not_fill(self):
        broker = PaperBroker(7.5, 6.0, 15.0)
        quiet = Candle(0, 100.0, 101.0, 99.0, 100.0, 10.0)
        self.assertIsNone(broker.stop_exit("X/USD", 10.0, 98.0, quiet, META,
                                           direction=1))
        self.assertIsNone(broker.stop_exit("X/USD", 10.0, 102.0, quiet, META,
                                           direction=-1))


class TestRestartWithThreeOpenPositions(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.broker = PaperBroker(7.5, 6.0, 15.0)
        self.acct = PaperAccount(self.repo, self.broker, 10_000.0, B,
                                 borrow_bps_per_day=15.0)
        self.sides = ["long", "short", "long"]
        for i, side in enumerate(self.sides):
            d = 1 if side == "long" else -1
            meta = MarketMeta(f"S{i}/USD", f"S{i}", "USD", True, 6, 6, 1e-6, 5.0)
            fill = self.broker.entry_fill(f"S{i}/USD", 5.0, 100.0, d, meta=meta)
            self.acct.open_position(
                symbol=f"S{i}/USD", strategy=B, strategy_version="1", qty=5.0,
                ref_price=100.0, fill=fill, initial_stop=100.0 - 2.0 * d,
                risk_amount=10.0, candle_id=f"c{i}", signal_score=70.0,
                journal={"ladder_slot": i + 1}, side=side)

    def test_three_positions_are_open(self):
        self.assertEqual(len(self.acct.positions()), 3)

    def test_all_three_survive_a_reopen_with_their_sides(self):
        cash = self.acct.cash()
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        got = repo2.get_positions(B)
        self.assertEqual(len(got), 3)
        self.assertEqual([p.side for p in got], self.sides)
        self.assertAlmostEqual(repo2.get_account(B)["cash"], cash, places=9)

    def test_collateral_and_slots_survive(self):
        margins = {p.symbol: p.margin_held for p in self.acct.positions()}
        slots = {p.symbol: p.journal.get("ladder_slot")
                 for p in self.acct.positions()}
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        for p in repo2.get_positions(B):
            self.assertAlmostEqual(p.margin_held, margins[p.symbol], places=9)
            self.assertEqual(p.journal.get("ladder_slot"), slots[p.symbol])

    def test_a_fourth_position_is_refused_after_the_restart(self):
        from crypto_edge.portfolio import ladder as L
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        n_open = len(repo2.get_positions(B))
        slot, _ = L.ceiling_for_slot(1_000.0, n_open, CFG.ladder_ceilings_pct)
        self.assertEqual(slot, 0, "the slot limit is a property of state, not memory")

    def test_strategy_a_is_untouched_by_any_of_it(self):
        self.assertEqual(self.repo.get_positions("trend_breakout"), [])
        self.assertEqual(self.repo.get_trades("trend_breakout"), [])


if __name__ == "__main__":
    unittest.main()


class RuntimeCase(unittest.TestCase):
    """Drive AggressiveRuntime against a fixture venue: entries, then exits."""

    def setUp(self):
        from unittest import mock

        from crypto_edge.aggressive_runtime import AggressiveRuntime
        from crypto_edge.config import Config
        from crypto_edge.research.journal import ResearchJournal
        from crypto_edge.strategy.base import MarketContext
        from fixtures_fast import frames

        self.symbols = [f"S{i:02d}/USD" for i in range(6)]
        self.frames = {s: frames((i - 2) * 0.00035, seed=700 + i, symbol=s,
                                 n_5m=6000)
                       for i, s in enumerate(self.symbols)}
        self.markets = {s: MarketMeta(s, s.split("/")[0], "USD", True,
                                      6, 6, 1e-6, 5.0) for s in self.symbols}
        outer = self

        class Feed:
            name, quote = "fixture", "USD"

            def fetch_ohlcv(self, symbol, timeframe, limit):
                return outer.frames[symbol][timeframe]

            def fetch_quote(self, symbol):
                from crypto_edge.models import TS_VENUE, Quote
                from crypto_edge.timeutils import now_ms
                px = float(outer.frames[symbol]["5m"].close[-1])
                return Quote(symbol, px * 0.9999, px * 1.0001, px,
                             now_ms(), TS_VENUE)

        self.repo, self.path = temp_repo()
        self.cfg = Config()
        self.cfg.telegram.enabled = False
        self.cfg.exchange.quote = "USD"
        self.cfg.resolve_symbols()
        self.broker = PaperBroker(7.5, 6.0, 15.0)
        self.rt = AggressiveRuntime(self.cfg, self.repo, Feed(),
                                    mock.MagicMock(), self.broker,
                                    ResearchJournal(self.repo))
        self.ctx = MarketContext(ts_ms=1_700_000_000_000, btc_regime="neutral",
                                 btc_regime_score=50.0, breadth_pct=50.0)
        self.hourly = {s: self.frames[s]["1h"] for s in self.symbols}
        self.meta_by = {s: {"dollar_volume": 4e7, "spread_bps": 4.0}
                        for s in self.symbols}

    def enter(self):
        return self.rt.scan_and_enter(
            self.ctx, rank_series=self.hourly, btc_1h=None,
            meta_by_symbol=self.meta_by, markets=self.markets)


class TestRuntimeEntries(RuntimeCase):
    def test_it_opens_positions_through_the_ladder(self):
        self.enter()
        self.assertGreater(len(self.rt.account.positions()), 0)

    def test_the_first_slot_takes_fifty_percent_times_the_multiplier(self):
        self.enter()
        first = min(self.rt.account.positions(), key=lambda p: p.entry_ms)
        j = first.journal
        self.assertEqual(j["ladder_slot"], 1)
        self.assertAlmostEqual(j["ladder_ceiling_cash"], 5_000.0, places=2)
        self.assertAlmostEqual(j["target_notional"],
                               5_000.0 * j["conf_multiplier"], places=2)

    def test_it_never_exceeds_three_positions(self):
        for _ in range(8):
            self.enter()
        self.assertLessEqual(len(self.rt.account.positions()),
                             self.cfg.aggressive.max_open_positions)

    def test_cash_never_goes_negative(self):
        for _ in range(8):
            self.enter()
        self.assertGreaterEqual(self.rt.account.cash(), 0.0)

    def test_every_entry_records_the_full_stage_three_journal(self):
        self.enter()
        required = ("setup_score", "confidence", "conf_bucket",
                    "conf_multiplier", "ladder_slot", "ladder_ceiling_cash",
                    "target_notional", "final_notional", "binding_constraint",
                    "stop_distance_pct", "expected_loss_cash",
                    "expected_loss_pct", "side", "borrow_bps_per_day",
                    "leverage", "max_loss_cash")
        for p in self.rt.account.positions():
            for key in required:
                self.assertIn(key, p.journal, f"{p.symbol} missing {key}")

    def test_the_expected_loss_respects_the_cap(self):
        self.enter()
        for p in self.rt.account.positions():
            self.assertLessEqual(p.journal["expected_loss_pct"],
                                 self.cfg.aggressive.max_loss_pct + 0.05,
                                 p.symbol)

    def test_confidence_equals_setup_score_in_phase_one(self):
        self.enter()
        for p in self.rt.account.positions():
            self.assertAlmostEqual(p.journal["confidence"],
                                   p.journal["setup_score"], places=9)

    def test_the_binding_constraint_is_recorded_and_counted(self):
        self.enter()
        self.assertTrue(self.rt.status.binding)
        for p in self.rt.account.positions():
            self.assertIn(p.journal["binding_constraint"],
                          ("ladder", "risk_cap", "exposure", "cash"))

    def test_rejected_candidates_are_journalled_too(self):
        self.enter()
        rows = self.repo.get_observations()
        self.assertGreater(len(rows), len(self.rt.account.positions()))
        self.assertTrue(any(r["decision"] != "ENTERED" for r in rows))

    def test_strategy_a_is_untouched_by_a_full_cycle(self):
        for _ in range(4):
            self.enter()
        self.assertEqual(self.repo.get_positions("trend_breakout"), [])
        self.assertEqual(self.repo.get_trades("trend_breakout"), [])


class TestRuntimeExits(RuntimeCase):
    def frames_by_symbol(self):
        return {s: self.frames[s] for s in self.symbols}

    def test_a_hostile_regime_closes_every_open_position(self):
        from crypto_edge.strategy.base import MarketContext
        self.enter()
        opened = len(self.rt.account.positions())
        self.assertGreater(opened, 0)
        hostile = MarketContext(ts_ms=self.ctx.ts_ms, btc_regime="bear",
                               btc_regime_score=10.0, breadth_pct=20.0)
        self.rt.manage(hostile, self.frames_by_symbol(), self.markets)
        self.assertEqual(len(self.rt.account.positions()), 0)
        self.assertEqual(self.rt.status.exits, opened)

    def test_the_exit_reason_is_recorded_on_the_trade(self):
        from crypto_edge.strategy.base import MarketContext
        self.enter()
        hostile = MarketContext(ts_ms=self.ctx.ts_ms, btc_regime="bear",
                               btc_regime_score=10.0, breadth_pct=20.0)
        self.rt.manage(hostile, self.frames_by_symbol(), self.markets)
        trades = self.repo.get_trades(B)
        self.assertTrue(trades)
        for t in trades:
            self.assertEqual(t["exit_reason"], "hostile_regime")
            self.assertIn(t["side"], ("long", "short"))

    def test_closing_frees_cash_and_slots(self):
        from crypto_edge.strategy.base import MarketContext
        self.enter()
        deployed_cash = self.rt.account.cash()
        hostile = MarketContext(ts_ms=self.ctx.ts_ms, btc_regime="bear",
                               btc_regime_score=10.0, breadth_pct=20.0)
        self.rt.manage(hostile, self.frames_by_symbol(), self.markets)
        self.assertGreater(self.rt.account.cash(), deployed_cash)
        self.assertEqual(len(self.rt.account.positions()), 0)

    def test_a_quiet_cycle_holds_the_positions(self):
        self.enter()
        n = len(self.rt.account.positions())
        self.rt.manage(self.ctx, self.frames_by_symbol(), self.markets)
        self.assertEqual(len(self.rt.account.positions()), n,
                         "nothing hostile happened; nothing should close")

    def test_round_trips_reconcile_cash_against_net_pnl(self):
        from crypto_edge.strategy.base import MarketContext
        start = self.rt.account.cash()
        self.enter()
        hostile = MarketContext(ts_ms=self.ctx.ts_ms, btc_regime="bear",
                               btc_regime_score=10.0, breadth_pct=20.0)
        self.rt.manage(hostile, self.frames_by_symbol(), self.markets)
        net = sum(t["net_pnl"] for t in self.repo.get_trades(B))
        self.assertAlmostEqual(self.rt.account.cash() - start, net, places=6)


class TestStageTwoFiltersAreUnchanged(unittest.TestCase):
    """Stage 3 sizes trades. It must not have moved a signal threshold."""

    def test_the_stage_two_gates_are_exactly_as_shipped(self):
        c = AggressiveCfg()
        self.assertEqual(c.min_setup_score, 50.0)
        self.assertEqual(c.min_rel_volume, 0.9)
        self.assertEqual(c.min_atr_pct, 0.25)
        self.assertEqual(c.min_ema_struct_15m, 0.5)
        self.assertEqual(c.max_hostile_ema_1h, -0.5)
        self.assertEqual(c.min_momentum_agree, 3)
        self.assertEqual(c.min_vote_atr, 0.15)
        self.assertEqual(c.shortlist_size, 12)

    def test_strategy_a_thresholds_are_exactly_as_shipped(self):
        from crypto_edge.config import StrategyCfg
        a = StrategyCfg()
        self.assertEqual(a.min_score, 55.0)
        self.assertEqual(a.donchian_lookback, 48)
        self.assertEqual(a.stop_atr_mult, 2.2)
        self.assertEqual(a.min_adx, 20.0)

    def test_stage_three_defaults_match_the_specification(self):
        c = AggressiveCfg()
        self.assertEqual(c.max_open_positions, 3)
        self.assertEqual(c.ladder_ceilings_pct, [50.0, 75.0, 100.0])
        self.assertEqual(c.max_loss_pct, 1.0)
        self.assertEqual(c.daily_buffer_fraction, 0.60)
        self.assertEqual(c.leverage, 1.0)
        self.assertEqual(c.min_confidence, 60.0)
        self.assertTrue(c.confidence_is_identity)


class TestStrategyBInsideTheEngineCycle(unittest.TestCase):
    """B is wired into the real cycle -- not merely importable beside it.

    The runtime tests above drive `AggressiveRuntime` directly. That proves the
    strategy works; it does not prove the ENGINE reaches it, and a pass that
    silently evaluates nothing looks exactly like a pass that found no setups.
    The distinction is not academic: run against `helpers.build_feed`, which
    holds 1h and 4h only, this path reports `evaluated: 0` with no error at all,
    because every 5m fetch raises DataUnavailable and is counted as a fetch
    failure. So these tests assert on the SCAN COUNTERS, not just on positions.
    """

    def setUp(self):
        from crypto_edge.engine import TradingEngine
        from crypto_edge.notify.telegram import TelegramNotifier
        from fixtures_fast import engine_feed
        from helpers import engine_config, temp_repo

        self.symbols = [f"S{i:02d}/USDT" for i in range(6)] + ["BTC/USDT"]
        self.feed, self.markets = engine_feed(self.symbols)
        self.repo, self.path = temp_repo()
        self.cfg = engine_config()
        self.cfg.universe.broad_static_assets = [s.split("/")[0]
                                                 for s in self.symbols]
        notifier = TelegramNotifier("t", "c", self.repo, enabled=False,
                                    transport=None, sleep=lambda _: None)
        self.engine = TradingEngine(self.cfg, self.repo, self.feed, notifier)

    def test_the_runtime_is_attached_when_the_strategy_is_enabled(self):
        self.assertIsNotNone(self.engine.aggressive)
        self.assertEqual(self.engine.aggressive.name, B)

    def test_no_runtime_is_attached_when_it_is_disabled(self):
        from crypto_edge.engine import TradingEngine
        from crypto_edge.notify.telegram import TelegramNotifier
        from helpers import temp_repo
        self.cfg.aggressive.enabled = False
        repo, _ = temp_repo()
        eng = TradingEngine(self.cfg, repo, self.feed,
                            TelegramNotifier("t", "c", repo, enabled=False,
                                             transport=None,
                                             sleep=lambda _: None))
        self.assertIsNone(eng.aggressive)

    def test_a_cycle_actually_evaluates_symbols(self):
        self.engine.cycle()
        st = self.engine.aggressive.status
        self.assertGreater(st.evaluated, 0,
                           "the engine reached the scan but it produced no "
                           "signals -- the feed cannot serve 5m/15m")
        self.assertGreater(st.deep_fetches, 0)

    def test_deep_fetches_stay_inside_the_shortlist_bound(self):
        self.engine.cycle()
        st = self.engine.aggressive.status
        # Two fast frames per shortlisted symbol, and never the whole universe.
        self.assertLessEqual(st.deep_fetches,
                             self.cfg.aggressive.shortlist_size * 2)

    def test_a_cycle_opens_positions_in_the_b_ledger(self):
        self.engine.cycle()
        rt = self.engine.aggressive
        self.assertGreater(len(rt.account.positions()), 0)
        for p in rt.account.positions():
            self.assertEqual(p.strategy, B)

    def test_the_two_ledgers_stay_separate_across_a_cycle(self):
        self.engine.cycle()
        a_syms = {p.symbol for p in self.engine.account.positions()}
        b_syms = {p.symbol for p in self.engine.aggressive.account.positions()}
        self.assertEqual(len(self.repo.get_positions(self.cfg.strategy.name)), len(a_syms))
        self.assertEqual(len(self.repo.get_positions(B)), len(b_syms))
        self.assertNotEqual(self.engine.account.cash(),
                            self.engine.aggressive.account.cash())

    def test_b_never_spends_more_than_its_own_ten_thousand(self):
        self.engine.cycle()
        rt = self.engine.aggressive
        self.assertGreaterEqual(rt.account.cash(), 0.0)
        # With no marks the equity is cash + posted collateral, so this is the
        # sub-account's own balance sheet: it cannot have grown by trading
        # against Strategy A's money.
        self.assertLessEqual(rt.equity({}), 10_000.0)

    def test_the_engine_fetches_the_ranking_timeframe(self):
        # Asserting this against the DEFAULT config proves nothing: B ranks on
        # 1h, which is already A's entry timeframe, so the union is unchanged
        # whether or not the engine adds it. Point B at a timeframe A does not
        # use, which is what the union exists for.
        from crypto_edge.engine import TradingEngine
        from crypto_edge.notify.telegram import TelegramNotifier
        from helpers import temp_repo
        self.cfg.aggressive.rank_timeframe = "30m"
        repo, _ = temp_repo()
        eng = TradingEngine(self.cfg, repo, self.feed,
                            TelegramNotifier("t", "c", repo, enabled=False,
                                             transport=None,
                                             sleep=lambda _: None))
        self.assertIn("30m", eng.timeframes())
        self.assertIn(self.cfg.strategy.entry_timeframe, eng.timeframes())
        self.assertIn(self.cfg.strategy.regime_timeframe, eng.timeframes())

    def test_the_shared_timeframe_list_never_repeats_one(self):
        # A's entry frame and B's ranking frame are both 1h by default: the
        # union must fetch it ONCE, not twice per symbol per cycle.
        tfs = self.engine.timeframes()
        self.assertEqual(len(tfs), len(set(tfs)))
        self.assertEqual(tfs.count(self.cfg.aggressive.rank_timeframe), 1)

    def test_the_fast_frames_are_not_fetched_for_the_whole_universe(self):
        # The two-phase scan exists to avoid exactly this.
        for tf in ("5m", "15m"):
            self.assertNotIn(tf, self.engine.timeframes())

    def test_a_faulty_strategy_b_never_stops_strategy_a(self):
        from unittest import mock
        with mock.patch.object(type(self.engine.aggressive), "scan_and_enter",
                               side_effect=RuntimeError("boom")):
            self.engine.cycle()          # must not raise
        self.assertEqual(len(self.engine.aggressive.account.positions()), 0)

    def test_entries_are_withheld_when_the_engine_says_so(self):
        # A's circuit breaker gates B's entries too -- but not B's scanning.
        # The counterfactual record is the point of the research journal, and a
        # halted day is exactly when it is most worth having.
        from unittest import mock
        rt = self.engine.aggressive
        with mock.patch.object(self.engine, "check_safety", return_value=False):
            self.engine.cycle()
        self.assertEqual(len(rt.account.positions()), 0)
        self.assertGreater(rt.status.evaluated, 0,
                           "a halted cycle should still SCAN and journal")

    def test_a_second_cycle_manages_the_positions_it_opened(self):
        self.engine.cycle()
        opened = {p.symbol for p in self.engine.aggressive.account.positions()}
        self.assertTrue(opened)
        self.engine.cycle()
        still = {p.symbol for p in self.engine.aggressive.account.positions()}
        closed = self.repo.get_trades(B)
        # Every position is accounted for: still open, or closed with a reason.
        self.assertEqual(opened - still, {t["symbol"] for t in closed} & opened)
        for t in closed:
            self.assertTrue(t["exit_reason"])
