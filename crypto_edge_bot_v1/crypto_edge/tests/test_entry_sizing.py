"""Final risk sizing must use the realistic entry fill.

DEFECT UNDER TEST
-----------------
`_enter()` sized the position on `sig.ref_price` -- the close of the last
CLOSED candle -- and then filled it at `quote.ask * (1 + slippage)`. Whenever
the ask had moved up since that close (the normal case for a breakout the
strategy just chased), the position was sized on the cheaper old price and
bought at the dearer new one, so:

    real risk = qty * (fill - stop)  >  qty * (ref - stop)  =  budget

The account quietly took more than `risk_per_trade_pct` on exactly the trades
where the market was moving fastest. Nothing reported it, because
`risk_amount` was recorded from the sizing call, not from the fill.

Corrected chain:

    ref_price --(quote ask + slippage + price rounding)--> expected fill
      --> size on the expected fill --> buy() --> revalidate on the real fill

The revalidation is the backstop: it recomputes risk from the fill that
actually happened and refuses the entry if it exceeds the budget.
"""
import unittest

import numpy as np

from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import MarketMeta, Quote
from crypto_edge.timeutils import now_ms
from helpers import breakout_closes, engine_config, trend_closes
from test_engine import build_engine

EQUITY = 10_000.0
RISK_PCT = 0.5
BUDGET = EQUITY * RISK_PCT / 100.0          # $50


def meta(amount_precision=8, price_precision=8, min_amount=0.0, min_cost=0.0):
    return MarketMeta("SOL/USDT", "SOL", "USDT", True, amount_precision,
                      price_precision, min_amount, min_cost)


def size_and_fill(b, ref, quote, m, *, equity=EQUITY, cash=1e9,
                  stop=95.0, risk_pct=RISK_PCT, max_position_pct=100.0,
                  max_exposure_pct=100.0, exposure=0.0):
    """Run the production chain: expected fill -> size -> buy -> revalidate."""
    est = b.expected_entry_price(ref, quote, m)
    sizing = b.size_position(
        equity=equity, cash=cash, entry_price=est, stop_price=stop,
        risk_pct=risk_pct, max_position_pct=max_position_pct,
        current_exposure=exposure, max_exposure_pct=max_exposure_pct, meta=m)
    if not sizing.ok:
        return sizing, None, None
    fill = b.buy("SOL/USDT", sizing.qty, ref, quote, m)
    ok, reason, actual_risk = b.revalidate_risk(
        sizing.qty, fill.fill_price, stop, equity, risk_pct)
    return sizing, fill, (ok, reason, actual_risk)


class TestExpectedFillMatchesTheRealFill(unittest.TestCase):
    """The whole fix rests on the prediction being exact, not approximate."""

    def test_prediction_equals_the_fill_across_many_configurations(self):
        for slip in (0.0, 6.0, 50.0):
            for use_book in (True, False):
                for price_precision in (2, 4, 8):
                    b = PaperBroker(7.5, slip, 15.0, use_book_spread=use_book)
                    m = meta(price_precision=price_precision)
                    q = Quote("SOL/USDT", 100.9, 101.1, 101.0, now_ms())
                    est = b.expected_entry_price(100.0, q, m)
                    fill = b.buy("SOL/USDT", 1.0, 100.0, q, m)
                    self.assertAlmostEqual(
                        est, fill.fill_price, places=12,
                        msg=f"slip={slip} book={use_book} prec={price_precision}")

    def test_prediction_handles_a_missing_quote_the_same_way_buy_does(self):
        b = PaperBroker(7.5, 6.0, 15.0, use_book_spread=True)
        m = meta(price_precision=4)
        self.assertAlmostEqual(b.expected_entry_price(100.0, None, m),
                               b.buy("SOL/USDT", 1.0, 100.0, None, m).fill_price,
                               places=12)


class TestSizingAgainstTheFill(unittest.TestCase):
    def setUp(self):
        self.b = PaperBroker(taker_fee_bps=7.5, slippage_bps=6.0,
                             stop_slippage_bps=15.0, use_book_spread=True)
        self.m = meta(amount_precision=4, price_precision=4)

    # --------------------------------------------------- quote == reference
    def test_quote_equal_to_reference_uses_the_full_budget(self):
        q = Quote("SOL/USDT", 100.0, 100.0, 100.0, now_ms())
        sizing, fill, (ok, reason, risk) = size_and_fill(self.b, 100.0, q, self.m)
        self.assertTrue(ok, reason)
        self.assertLessEqual(risk, BUDGET * 1.01)
        self.assertGreater(risk, BUDGET * 0.95, "budget should be nearly fully used")

    # ---------------------------------------------------- quote > reference
    def test_quote_above_reference_does_not_inflate_risk(self):
        """The exact defect: ask has run 2% above the signal candle's close."""
        q = Quote("SOL/USDT", 101.9, 102.0, 101.95, now_ms())
        sizing, fill, (ok, reason, risk) = size_and_fill(self.b, 100.0, q, self.m)
        self.assertTrue(ok, reason)
        self.assertGreater(fill.fill_price, 100.0, "precondition: we filled higher")
        self.assertLessEqual(risk, BUDGET * 1.01,
                             "risk must stay inside the configured budget")

    def test_the_old_reference_priced_sizing_would_have_overshot(self):
        """Demonstrates the bug is real, not theoretical."""
        q = Quote("SOL/USDT", 101.9, 102.0, 101.95, now_ms())
        stop = 95.0
        old_qty = self.b.size_position(
            equity=EQUITY, cash=1e9, entry_price=100.0, stop_price=stop,
            risk_pct=RISK_PCT, max_position_pct=100.0, current_exposure=0.0,
            max_exposure_pct=100.0, meta=self.m).qty
        fill = self.b.buy("SOL/USDT", old_qty, 100.0, q, self.m)
        old_risk = old_qty * (fill.fill_price - stop)
        self.assertGreater(old_risk, BUDGET * 1.02,
                           "the old code really did exceed the risk budget")

        _, new_fill, (ok, _, new_risk) = size_and_fill(self.b, 100.0, q, self.m)
        self.assertLess(new_risk, old_risk)
        self.assertLessEqual(new_risk, BUDGET * 1.01)

    def test_large_quote_movement_shrinks_the_position_further(self):
        stop = 95.0
        risks, qtys = [], []
        for ask in (100.0, 102.0, 105.0, 110.0):
            q = Quote("SOL/USDT", ask - 0.05, ask, ask, now_ms())
            sizing, fill, (ok, reason, risk) = size_and_fill(
                self.b, 100.0, q, self.m, stop=stop)
            self.assertTrue(ok, f"ask={ask}: {reason}")
            self.assertLessEqual(risk, BUDGET * 1.01, f"ask={ask}")
            risks.append(risk)
            qtys.append(sizing.qty)
        self.assertTrue(all(a >= b for a, b in zip(qtys, qtys[1:])),
                        "a higher ask must never buy MORE units")

    def test_quote_running_through_the_stop_is_refused_outright(self):
        """If the market has already moved past where the stop would sit, there
        is no position to size -- not a smaller one."""
        q = Quote("SOL/USDT", 94.0, 94.5, 94.2, now_ms())
        sizing, fill, res = size_and_fill(self.b, 100.0, q, self.m, stop=95.0)
        self.assertFalse(sizing.ok)
        self.assertIn("stop must be below entry", sizing.reason)

    # ------------------------------------------------------------ slippage
    def test_slippage_is_priced_into_the_size(self):
        q = Quote("SOL/USDT", 99.99, 100.0, 100.0, now_ms())
        heavy = PaperBroker(7.5, slippage_bps=200.0, stop_slippage_bps=15.0)
        light = PaperBroker(7.5, slippage_bps=0.0, stop_slippage_bps=15.0)
        s_heavy, f_heavy, (ok_h, _, r_heavy) = size_and_fill(heavy, 100.0, q, self.m)
        s_light, f_light, (ok_l, _, r_light) = size_and_fill(light, 100.0, q, self.m)
        self.assertTrue(ok_h and ok_l)
        self.assertGreater(f_heavy.fill_price, f_light.fill_price)
        self.assertLess(s_heavy.qty, s_light.qty,
                        "more modelled slippage must buy fewer units")
        self.assertLessEqual(r_heavy, BUDGET * 1.01)

    # ------------------------------------------------------------ rounding
    def test_amount_rounding_only_ever_rounds_risk_down(self):
        for precision in (0, 1, 2, 4, 8):
            m = meta(amount_precision=precision, price_precision=4)
            q = Quote("SOL/USDT", 100.9, 101.0, 100.95, now_ms())
            sizing, fill, (ok, reason, risk) = size_and_fill(self.b, 100.0, q, m)
            if not sizing.ok:
                continue
            self.assertTrue(ok, f"precision={precision}: {reason}")
            self.assertLessEqual(risk, BUDGET * 1.01, f"precision={precision}")
            self.assertEqual(sizing.qty, float(np.floor(sizing.qty * 10 ** precision)
                                               / 10 ** precision))

    def test_price_rounding_is_included_in_the_sized_price(self):
        m = meta(amount_precision=4, price_precision=0)   # whole-dollar ticks
        q = Quote("SOL/USDT", 100.4, 100.6, 100.5, now_ms())
        sizing, fill, (ok, reason, risk) = size_and_fill(self.b, 100.0, q, m)
        self.assertTrue(ok, reason)
        self.assertEqual(sizing.entry_price, fill.fill_price)
        self.assertLessEqual(risk, BUDGET * 1.01)

    # ----------------------------------------------------- max allocation
    def test_max_position_cap_binds_before_the_risk_budget(self):
        q = Quote("SOL/USDT", 100.9, 101.0, 100.95, now_ms())
        # a far stop would want a huge notional; the allocation cap must win
        sizing, fill, (ok, reason, risk) = size_and_fill(
            self.b, 100.0, q, self.m, stop=50.0, max_position_pct=5.0)
        self.assertTrue(ok, reason)
        self.assertLessEqual(sizing.notional, EQUITY * 5.0 / 100.0 + 1e-6)
        self.assertLess(risk, BUDGET, "capped allocation must risk LESS, not more")

    def test_exposure_cap_and_fill_price_interact_correctly(self):
        q = Quote("SOL/USDT", 100.9, 101.0, 100.95, now_ms())
        sizing, fill, (ok, reason, risk) = size_and_fill(
            self.b, 100.0, q, self.m, exposure=5_800.0, max_exposure_pct=60.0)
        self.assertTrue(ok, reason)
        # room = 60% of 10k - 5800 = 200, valued at the FILL price
        self.assertLessEqual(sizing.notional, 200.0 + 1e-6)
        self.assertLessEqual(risk, BUDGET * 1.01)

    def test_exhausted_exposure_is_rejected(self):
        q = Quote("SOL/USDT", 100.9, 101.0, 100.95, now_ms())
        sizing, _, _ = size_and_fill(self.b, 100.0, q, self.m,
                                     exposure=6_000.0, max_exposure_pct=60.0)
        self.assertFalse(sizing.ok)
        self.assertIn("exposure limit", sizing.reason)

    # ------------------------------------------------------------ fees
    def test_cash_cap_accounts_for_the_entry_fee_at_the_fill_price(self):
        q = Quote("SOL/USDT", 100.9, 101.0, 100.95, now_ms())
        cash = 500.0
        sizing, fill, (ok, reason, risk) = size_and_fill(
            self.b, 100.0, q, self.m, cash=cash, stop=50.0)
        self.assertTrue(ok, reason)
        self.assertLessEqual(fill.notional + fill.fee, cash + 1e-6,
                             "notional plus fee must fit the cash we actually have")

    def test_fees_never_push_the_position_over_the_risk_budget(self):
        q = Quote("SOL/USDT", 100.9, 101.0, 100.95, now_ms())
        for fee_bps in (0.0, 7.5, 50.0, 200.0):
            b = PaperBroker(fee_bps, 6.0, 15.0, use_book_spread=True)
            sizing, fill, (ok, reason, risk) = size_and_fill(b, 100.0, q, self.m)
            self.assertTrue(ok, f"fee={fee_bps}: {reason}")
            self.assertLessEqual(risk, BUDGET * 1.01, f"fee={fee_bps}")

    def test_true_cost_at_stop_including_fees_is_reported(self):
        """Price risk is the sized quantity; fees and stop slippage are real
        money on top. We do not hide them -- we report them."""
        q = Quote("SOL/USDT", 100.0, 100.0, 100.0, now_ms())
        sizing, fill, (ok, reason, risk) = size_and_fill(self.b, 100.0, q, self.m)
        self.assertTrue(ok, reason)
        self.assertGreater(sizing.est_cost_at_stop, sizing.risk_amount,
                           "the all-in cost of a stop-out exceeds price risk alone")
        self.assertLess(sizing.est_cost_at_stop, sizing.risk_amount * 1.5,
                        "but it should be the same order of magnitude")


class TestRevalidationBackstop(unittest.TestCase):
    def setUp(self):
        self.b = PaperBroker(7.5, 6.0, 15.0)

    def test_oversized_position_is_refused(self):
        qty_for_double_risk = 2 * BUDGET / (100.0 - 95.0)
        ok, reason, risk = self.b.revalidate_risk(
            qty_for_double_risk, 100.0, 95.0, EQUITY, RISK_PCT)
        self.assertFalse(ok)
        self.assertIn("exceeds budget", reason)
        self.assertAlmostEqual(risk, 2 * BUDGET, places=6)

    def test_correctly_sized_position_passes(self):
        ok, reason, risk = self.b.revalidate_risk(
            BUDGET / 5.0, 100.0, 95.0, EQUITY, RISK_PCT)
        self.assertTrue(ok, reason)
        self.assertAlmostEqual(risk, BUDGET, places=6)

    def test_tolerance_absorbs_float_noise_but_not_real_drift(self):
        qty = BUDGET / 5.0
        self.assertTrue(self.b.revalidate_risk(
            qty * 1.005, 100.0, 95.0, EQUITY, RISK_PCT, tolerance_pct=1.0)[0])
        self.assertFalse(self.b.revalidate_risk(
            qty * 1.10, 100.0, 95.0, EQUITY, RISK_PCT, tolerance_pct=1.0)[0])

    def test_stop_at_or_above_the_fill_is_refused(self):
        for stop in (100.0, 105.0):
            ok, reason, _ = self.b.revalidate_risk(1.0, 100.0, stop, EQUITY, RISK_PCT)
            self.assertFalse(ok)
            self.assertIn("not below the simulated fill", reason)

    def test_degenerate_inputs_are_refused(self):
        self.assertFalse(self.b.revalidate_risk(0.0, 100.0, 95.0, EQUITY, RISK_PCT)[0])
        self.assertFalse(self.b.revalidate_risk(1.0, 0.0, 95.0, EQUITY, RISK_PCT)[0])
        self.assertFalse(self.b.revalidate_risk(
            float("nan"), 100.0, 95.0, EQUITY, RISK_PCT)[0])


class TestEngineEntryRiskEndToEnd(unittest.TestCase):
    """The engine, not just the broker, must honour the budget."""

    def setUp(self):
        self.closes = {"BTC/USDT": trend_closes(400, seed=4),
                       "SOL/USDT": breakout_closes(400, seed=11)}

    def _enter_with_ask_multiplier(self, mult):
        cfg = engine_config()
        cfg.execution.use_book_spread = True
        engine, feed, transport, repo = build_engine(self.closes, cfg=cfg)
        base_quote = feed.fetch_quote

        def moved(symbol):
            q = base_quote(symbol)
            if q is None:
                return None
            return type(q)(q.symbol, q.bid * mult, q.ask * mult,
                           q.last * mult, q.ts_ms)

        feed.fetch_quote = moved
        engine.cycle()
        return engine, repo

    def test_position_risk_stays_within_budget_when_the_ask_has_run_up(self):
        engine, repo = self._enter_with_ask_multiplier(1.02)
        positions = engine.account.positions()
        self.assertEqual(len(positions), 1, "a 2% move should still be tradable")
        p = positions[0]
        equity = float(engine.account.state["starting_equity"])
        budget = equity * engine.cfg.risk.risk_per_trade_pct / 100.0
        realised_risk = p.qty * (p.entry_fill_price - p.initial_stop)
        self.assertLessEqual(realised_risk, budget * 1.01)
        self.assertAlmostEqual(p.risk_amount, realised_risk, places=6,
                               msg="the stored risk_amount must be the fill-based one")

    def test_risk_stays_bounded_across_a_range_of_quote_moves(self):
        for mult in (1.0, 1.005, 1.01, 1.02, 1.05):
            engine, repo = self._enter_with_ask_multiplier(mult)
            positions = engine.account.positions()
            if not positions:
                continue          # correctly refused; nothing to check
            p = positions[0]
            budget = 10_000.0 * engine.cfg.risk.risk_per_trade_pct / 100.0
            risk = p.qty * (p.entry_fill_price - p.initial_stop)
            self.assertLessEqual(risk, budget * 1.01, f"ask multiplier {mult}")

    def test_the_journal_records_both_prices_and_the_budget(self):
        engine, repo = self._enter_with_ask_multiplier(1.02)
        p = engine.account.positions()[0]
        j = p.journal
        for key in ("entry_ref_price", "expected_fill_price", "entry_fill_price",
                    "risk_budget", "risk_at_fill", "est_cost_at_stop"):
            self.assertIn(key, j, f"{key} must be journalled for later audit")
        self.assertGreater(j["entry_fill_price"], j["entry_ref_price"])
        self.assertAlmostEqual(j["expected_fill_price"], j["entry_fill_price"],
                               places=9)
        self.assertLessEqual(j["risk_at_fill"], j["risk_budget"] * 1.01)


if __name__ == "__main__":
    unittest.main()
