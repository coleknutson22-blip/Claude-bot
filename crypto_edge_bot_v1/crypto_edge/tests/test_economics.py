"""Accounting basis, fee tiers and ATR economics.

WHAT THESE TESTS ARE DEFENDING
------------------------------
This module exists because a conclusion was drawn from mixed accounting bases
and had to be withdrawn. The failure mode is silent: `gross_pnl` (before fees
and slippage) and `trading["gross_profit"]` (sum of NET P&L over net-winners)
are both called "gross" in the same report, so combining a win count from one
with a total from the other produces arithmetic that looks fine and means
nothing. Four guards:

  1. A BASIS THAT DOES NOT TRAVEL WITH ITS NUMBER. Every statistic here is
     computed from ONE list of per-trade figures, and carries the basis in the
     data. The tests assert that a trade which is a gross win and a net loss
     lands on opposite sides of the two partitions.

  2. A FEE CHANGE THAT MOVES SOMETHING ELSE. Re-pricing must hold fills,
     slippage and gross P&L exactly; only the fee may move. Anything else
     smuggles a second variable into a one-variable comparison.

  3. A TIER THAT IS SELECTED AND THEN NOT USED. `effective_taker_bps` must
     win over the raw field everywhere, or fills price at one rate and reports
     quote another.

  4. AN "UNTRADEABLE" CELL THAT IS MERELY EXPENSIVE. A setup is structurally
     untradeable when a PERFECT trade still loses money -- not when its
     break-even win rate is high. The two need different answers.

All four are tested directly and mutation-tested.
"""
import math
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import (FEE_TIER_CUSTOM, KRAKEN_SPOT_TAKER_BPS, Config,
                                fee_tier_bps)
from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import ClosedTrade
from crypto_edge.research import economics as ec
from helpers import temp_repo

B = "aggressive_momentum_v2"
T0 = 1_700_000_000_000


class Base(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.repo, _ = temp_repo()
        self.i = 0

    def trade(self, *, gross, fees, slippage, financing=0.0, qty=25.0,
              entry_fill=100.0, exit_fill=100.0, initial_stop=98.2,
              reason="stop", side="long"):
        """A ledger row with the P&L decomposition stated explicitly.

        Written from the components rather than derived from prices so a test
        can place a trade precisely on either side of the gross/net divide,
        which is the whole subject here.
        """
        self.i += 1
        d = -1 if side == "short" else 1
        entry_ref = entry_fill
        exit_ref = entry_ref + gross / (qty * d)
        self.repo.add_trade(ClosedTrade(
            id=f"trd{self.i}", position_id=f"pos{self.i}",
            symbol=f"S{self.i}/USD", strategy=B, strategy_version="v2",
            side=side, qty=qty, entry_ref_price=entry_ref,
            entry_fill_price=entry_fill, entry_ms=T0 + self.i * 600_000,
            exit_ref_price=exit_ref, exit_fill_price=exit_fill,
            exit_ms=T0 + self.i * 600_000 + 3_600_000, exit_reason=reason,
            initial_stop=initial_stop, final_stop=initial_stop,
            gross_pnl=gross, fees=fees, slippage_cost=slippage,
            net_pnl=gross - fees - slippage - financing, financing=financing,
            return_pct=0.0, account_return_pct=0.0, mfe=1.0, mae=-1.0,
            duration_s=3600.0, equity_after=10_000.0, journal={}))


# ====================================== 1. the two bases must stay apart
class TestAccountingBasis(Base):
    def test_a_trade_can_be_a_gross_win_and_a_net_loss(self):
        # THE defect. +$5 of market move against $11.50 of costs.
        self.trade(gross=5.0, fees=4.0, slippage=7.5)
        q = ec.exit_quality(self.repo, B)
        self.assertEqual(q["gross"]["wins"], 1)
        self.assertEqual(q["gross"]["losses"], 0)
        self.assertEqual(q["net"]["wins"], 0)
        self.assertEqual(q["net"]["losses"], 1)
        self.assertEqual(q["cost_flipped"], 1)
        self.assertAlmostEqual(q["cost_flipped_gross"], 5.0, places=9)

    def test_the_win_rates_differ_when_costs_flip_a_trade(self):
        self.trade(gross=5.0, fees=4.0, slippage=7.5)     # flips
        self.trade(gross=50.0, fees=4.0, slippage=7.5)    # wins on both
        self.trade(gross=-20.0, fees=4.0, slippage=7.5)   # loses on both
        q = ec.exit_quality(self.repo, B)
        self.assertAlmostEqual(q["gross"]["win_rate_pct"], 200 / 3.0, places=6)
        self.assertAlmostEqual(q["net"]["win_rate_pct"], 100 / 3.0, places=6)

    def test_each_basis_totals_its_own_partition(self):
        self.trade(gross=5.0, fees=4.0, slippage=7.5)
        self.trade(gross=50.0, fees=4.0, slippage=7.5)
        q = ec.exit_quality(self.repo, B)
        self.assertAlmostEqual(q["gross"]["total"], 55.0, places=9)
        self.assertAlmostEqual(q["net"]["total"], 55.0 - 8.0 - 15.0, places=9)
        # The gross average winner is over BOTH trades; the net one over one.
        self.assertAlmostEqual(q["gross"]["avg_win"], 27.5, places=9)
        self.assertAlmostEqual(q["net"]["avg_win"], 50.0 - 11.5, places=9)

    def test_every_statistic_carries_its_own_basis(self):
        self.trade(gross=5.0, fees=4.0, slippage=7.5)
        q = ec.exit_quality(self.repo, B)
        self.assertEqual(q["gross"]["basis"], ec.GROSS)
        self.assertEqual(q["net"]["basis"], ec.NET)
        self.assertEqual(q["r_gross"]["basis"], ec.GROSS)
        self.assertEqual(q["r_net"]["basis"], ec.NET)

    def test_a_net_above_gross_is_flagged_as_impossible(self):
        # The cost model cannot produce it. If the ledger ever shows one, the
        # ledger is wrong and nothing computed from it can be trusted.
        self.trade(gross=1.0, fees=-5.0, slippage=0.0)
        self.assertEqual(ec.exit_quality(self.repo, B)["impossible_rows"], 1)

    def test_a_zero_gross_trade_is_not_cost_flipped(self):
        # Flipped means the MARKET paid and the costs took it back. A trade
        # that made nothing gross had nothing taken back, and counting it
        # would inflate the very figure that measures the mix-up.
        self.trade(gross=0.0, fees=4.0, slippage=7.5)
        q = ec.exit_quality(self.repo, B)
        self.assertEqual(q["cost_flipped"], 0)
        self.assertEqual(q["gross"]["wins"], 0)
        self.assertEqual(q["net"]["wins"], 0)

    def test_no_flipped_trades_when_costs_are_zero(self):
        self.trade(gross=5.0, fees=0.0, slippage=0.0)
        q = ec.exit_quality(self.repo, B)
        self.assertEqual(q["cost_flipped"], 0)
        self.assertEqual(q["gross"]["wins"], q["net"]["wins"])

    def test_profit_factor_is_computed_within_one_basis(self):
        self.trade(gross=30.0, fees=0.0, slippage=0.0)
        self.trade(gross=-10.0, fees=0.0, slippage=0.0)
        q = ec.exit_quality(self.repo, B)
        self.assertAlmostEqual(q["gross"]["profit_factor"], 3.0, places=9)

    def test_profit_factor_is_infinite_with_no_losses_and_none_with_no_trades(self):
        self.assertIsNone(ec.basis_stats([], ec.GROSS).profit_factor)
        self.assertEqual(ec.basis_stats([5.0], ec.GROSS).profit_factor,
                         float("inf"))

    def test_a_breakeven_trade_counts_as_a_loss_not_a_win(self):
        # `> 0` on both sides, so zero never inflates a win rate.
        self.assertEqual(ec.basis_stats([0.0], ec.NET).wins, 0)
        self.assertEqual(ec.basis_stats([0.0], ec.NET).losses, 1)


# ================================================== 2. R-multiples
class TestRMultiples(Base):
    def test_r_is_measured_against_the_initial_stop(self):
        # entry 100, initial stop 98 -> 2.00 of risk per unit, 25 units = 50.
        self.trade(gross=100.0, fees=0.0, slippage=0.0, qty=25.0,
                   entry_fill=100.0, initial_stop=98.0)
        q = ec.exit_quality(self.repo, B)
        self.assertAlmostEqual(q["r_gross"]["avg_win"], 2.0, places=9)

    def test_a_short_stop_above_entry_still_gives_positive_risk(self):
        self.trade(gross=50.0, fees=0.0, slippage=0.0, qty=25.0,
                   entry_fill=100.0, initial_stop=102.0, side="short")
        self.assertAlmostEqual(
            ec.exit_quality(self.repo, B)["r_gross"]["avg_win"], 1.0, places=9)

    def test_r_ignores_a_ratcheted_stop(self):
        # The trail is not the risk that was taken. Dividing by a stop that
        # was moved to breakeven makes every trailed winner an enormous
        # multiple of a risk nobody ever had on.
        self.i += 1
        from crypto_edge.models import ClosedTrade
        self.repo.add_trade(ClosedTrade(
            id="rt1", position_id="p1", symbol="S/USD", strategy=B,
            strategy_version="v2", side="long", qty=25.0,
            entry_ref_price=100.0, entry_fill_price=100.0, entry_ms=T0,
            exit_ref_price=104.0, exit_fill_price=104.0,
            exit_ms=T0 + 3_600_000, exit_reason="target",
            initial_stop=98.0,        # 2.00/unit -> 50.00 of risk
            final_stop=100.0,         # ratcheted to breakeven -> 0 of "risk"
            gross_pnl=100.0, fees=0.0, slippage_cost=0.0, net_pnl=100.0,
            financing=0.0, return_pct=0.0, account_return_pct=0.0,
            mfe=1.0, mae=-1.0, duration_s=3600.0, equity_after=10_100.0,
            journal={}))
        q = ec.exit_quality(self.repo, B)
        self.assertEqual(q["r_measurable"], 1)
        self.assertAlmostEqual(q["r_gross"]["avg_win"], 2.0, places=9)

    def test_initial_risk_refuses_a_zero_distance(self):
        # Guarded at the source, not only by the callers' truthiness checks:
        # a zero returned here would divide by zero the moment a caller
        # stopped checking.
        self.trade(gross=10.0, fees=0.0, slippage=0.0, entry_fill=100.0,
                   initial_stop=100.0)
        t = self.repo.get_trades(B)[0]
        self.assertIsNone(ec.initial_risk(t))

    def test_a_stop_at_the_entry_is_unmeasurable_not_infinite(self):
        self.trade(gross=10.0, fees=0.0, slippage=0.0, entry_fill=100.0,
                   initial_stop=100.0)
        q = ec.exit_quality(self.repo, B)
        self.assertEqual(q["r_measurable"], 0)
        self.assertEqual(q["r_unmeasurable"], 1)

    def test_gross_and_net_r_are_both_reported(self):
        self.trade(gross=100.0, fees=10.0, slippage=15.0, qty=25.0,
                   entry_fill=100.0, initial_stop=98.0)
        q = ec.exit_quality(self.repo, B)
        self.assertAlmostEqual(q["r_gross"]["expectancy"], 2.0, places=9)
        self.assertAlmostEqual(q["r_net"]["expectancy"], 75.0 / 50.0, places=9)


# ================================================== 3. fee tiers in config
class TestFeeTiers(unittest.TestCase):
    def test_the_default_changes_nothing(self):
        x = Config().execution
        self.assertEqual(x.fee_tier, FEE_TIER_CUSTOM)
        self.assertEqual(x.effective_taker_bps(), x.taker_fee_bps)

    def test_a_selected_tier_wins_over_the_raw_field(self):
        # The raw field must NOT leak through, or fills price at one rate and
        # reports quote another -- which makes a strategy look viable at a fee
        # it is not paying.
        x = Config().execution
        x.fee_tier = "tier1"
        self.assertEqual(x.effective_taker_bps(), 80.0)
        self.assertNotEqual(x.effective_taker_bps(), x.taker_fee_bps)

    def test_the_custom_tier_returns_the_rate_it_was_given(self):
        # Not a hardcoded 7.5. A config that edits `taker_fee_bps` without
        # naming a tier must still be priced at what it says.
        self.assertEqual(fee_tier_bps(FEE_TIER_CUSTOM, 42.0), 42.0)
        x = Config().execution
        x.taker_fee_bps = 12.25
        self.assertEqual(x.effective_taker_bps(), 12.25)
        self.assertIn("12.25", x.fee_label())

    def test_every_published_tier_resolves(self):
        for name, bps in KRAKEN_SPOT_TAKER_BPS.items():
            self.assertEqual(fee_tier_bps(name, 7.5), bps)

    def test_the_tier_ladder_is_monotonic_in_volume(self):
        # Tier 1 is the smallest account and pays the most. A table that ever
        # inverts would make every scenario read backwards.
        order = ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6"]
        bps = [KRAKEN_SPOT_TAKER_BPS[t] for t in order]
        self.assertEqual(bps, sorted(bps, reverse=True))

    def test_an_unknown_tier_is_rejected_by_validation(self):
        # Fail closed at startup. A typo'd tier that silently fell back to the
        # raw field would price every fill at 7.5 bps while the operator
        # believed they were simulating 80.
        cfg = Config()
        cfg.execution.fee_tier = "platinum"
        errs = cfg.validate()
        self.assertTrue(any("fee_tier" in e for e in errs), errs)
        self.assertTrue(any("platinum" in e for e in errs), errs)

    def test_a_valid_config_reports_no_fee_tier_error(self):
        for tier in [FEE_TIER_CUSTOM, *KRAKEN_SPOT_TAKER_BPS]:
            cfg = Config()
            cfg.execution.fee_tier = tier
            self.assertFalse([e for e in cfg.validate() if "fee_tier" in e],
                             tier)

    def test_the_label_names_the_tier_being_simulated(self):
        x = Config().execution
        self.assertIn("custom", x.fee_label())
        x.fee_tier = "tier3"
        self.assertIn("tier3", x.fee_label())
        self.assertIn("38", x.fee_label())

    def test_the_broker_charges_the_selected_tier(self):
        # End to end: the tier must reach an actual fill, not just a report.
        x = Config().execution
        x.fee_tier = "tier1"
        broker = PaperBroker(x.effective_taker_bps(), x.slippage_bps,
                             x.stop_slippage_bps)
        self.assertAlmostEqual(broker.fee(10_000.0), 80.0, places=9)


# ============================================== 4. re-pricing the ledger
class TestFeeScenarios(Base):
    def test_only_the_fee_moves(self):
        self.trade(gross=40.0, fees=10.0, slippage=15.0, financing=2.0,
                   qty=25.0, entry_fill=100.0, exit_fill=100.0)
        t = self.repo.get_trades(B)[0]
        p = ec.refee_trade(t, 50.0)
        self.assertAlmostEqual(p["gross_pnl"], 40.0, places=9)
        self.assertAlmostEqual(p["slippage_cost"], 15.0, places=9)
        self.assertAlmostEqual(p["financing"], 2.0, places=9)

    def test_fees_are_charged_on_both_legs(self):
        self.trade(gross=0.0, fees=0.0, slippage=0.0, qty=25.0,
                   entry_fill=100.0, exit_fill=100.0)
        t = self.repo.get_trades(B)[0]
        p = ec.refee_trade(t, 50.0)      # 50 bps of 2 x 2500 = 25.00
        self.assertAlmostEqual(p["two_leg_notional"], 5000.0, places=9)
        self.assertAlmostEqual(p["fees"], 25.0, places=9)

    def test_legs_are_priced_on_their_own_notionals(self):
        # Halving the exit price halves the exit leg's fee. A model that used
        # the entry notional twice would overcharge every trade that moved.
        self.trade(gross=0.0, fees=0.0, slippage=0.0, qty=25.0,
                   entry_fill=100.0, exit_fill=50.0)
        p = ec.refee_trade(self.repo.get_trades(B)[0], 100.0)
        self.assertAlmostEqual(p["two_leg_notional"], 3750.0, places=9)
        self.assertAlmostEqual(p["fees"], 37.5, places=9)

    def test_the_configured_rate_reproduces_the_recorded_fees(self):
        # Sanity that the re-pricing is the same arithmetic the ledger used.
        qty, px = 25.0, 100.0
        fees = 2 * qty * px * 7.5 / 10_000.0
        self.trade(gross=0.0, fees=fees, slippage=0.0, qty=qty,
                   entry_fill=px, exit_fill=px)
        p = ec.refee_trade(self.repo.get_trades(B)[0], 7.5)
        self.assertAlmostEqual(p["fees"], fees, places=9)
        self.assertAlmostEqual(p["net_pnl"], -fees, places=9)

    def test_every_tier_appears_and_the_configured_one_is_included(self):
        self.trade(gross=10.0, fees=1.0, slippage=1.0)
        out = ec.fee_scenarios(self.repo, B, 7.5)
        names = [r["tier"] for r in out["rows"]]
        self.assertEqual(names[0], FEE_TIER_CUSTOM)
        for t in KRAKEN_SPOT_TAKER_BPS:
            self.assertIn(t, names)

    def test_a_higher_tier_is_strictly_worse(self):
        self.trade(gross=100.0, fees=1.0, slippage=1.0, qty=25.0,
                   entry_fill=100.0, exit_fill=100.0)
        rows = {r["tier"]: r for r in ec.fee_scenarios(self.repo, B, 7.5)["rows"]}
        self.assertGreater(rows["tier6"]["net_pnl"], rows["tier1"]["net_pnl"])
        self.assertGreater(rows["tier1"]["total_fees"], rows["tier6"]["total_fees"])

    def test_gross_and_slippage_are_identical_across_every_tier(self):
        self.trade(gross=100.0, fees=1.0, slippage=7.0, qty=25.0)
        out = ec.fee_scenarios(self.repo, B, 7.5)
        self.assertEqual({r["gross_pnl"] for r in out["rows"]}, {100.0})
        self.assertEqual({r["slippage"] for r in out["rows"]}, {7.0})

    def test_breakeven_gross_is_the_total_cost_at_that_tier(self):
        self.trade(gross=5.0, fees=1.0, slippage=7.0, financing=2.0, qty=25.0,
                   entry_fill=100.0, exit_fill=100.0)
        rows = {r["tier"]: r for r in ec.fee_scenarios(self.repo, B, 7.5)["rows"]}
        r = rows["tier1"]
        self.assertAlmostEqual(r["breakeven_gross_required"],
                               r["total_fees"] + 7.0 + 2.0, places=9)
        # At break-even gross, net would be exactly zero.
        self.assertAlmostEqual(
            r["breakeven_gross_required"] - r["total_fees"] - 7.0 - 2.0,
            0.0, places=9)

    def test_the_shortfall_multiple_is_withheld_without_a_positive_edge(self):
        # A negative gross edge is not "nearly there": no fee tier fixes a sign.
        self.trade(gross=-50.0, fees=1.0, slippage=1.0)
        for r in ec.fee_scenarios(self.repo, B, 7.5)["rows"]:
            self.assertIsNone(r["gross_shortfall_multiple"])
            self.assertFalse(r["viable"])

    def test_the_shortfall_multiple_is_reported_with_a_positive_edge(self):
        self.trade(gross=50.0, fees=1.0, slippage=1.0, qty=25.0,
                   entry_fill=100.0, exit_fill=100.0)
        rows = {r["tier"]: r for r in ec.fee_scenarios(self.repo, B, 7.5)["rows"]}
        m = rows["tier1"]["gross_shortfall_multiple"]
        self.assertIsNotNone(m)
        self.assertAlmostEqual(
            m, rows["tier1"]["breakeven_gross_required"] / 50.0, places=9)

    def test_viability_follows_net_not_gross(self):
        # A POSITIVE gross edge that survives 7.5 bps and dies at 80. Reading
        # viability off gross would call every tier profitable, which is
        # exactly the conclusion this whole exercise is testing.
        # 2 x 2500 notional: 7.5 bps = $3.75, 80 bps = $40.00.
        self.trade(gross=20.0, fees=3.75, slippage=1.0, qty=25.0,
                   entry_fill=100.0, exit_fill=100.0)
        rows = {r["tier"]: r for r in ec.fee_scenarios(self.repo, B, 7.5)["rows"]}
        self.assertGreater(rows["custom"]["net_pnl"], 0.0)
        self.assertLess(rows["tier1"]["net_pnl"], 0.0)
        self.assertTrue(rows["custom"]["viable"])
        self.assertFalse(rows["tier1"]["viable"])
        # And the crossover sits between them, not at the extremes.
        flips = [r["tier"] for r in ec.fee_scenarios(self.repo, B, 7.5)["rows"]
                 if r["viable"]]
        self.assertIn("custom", flips)
        self.assertNotIn("tier1", flips)

    def test_an_empty_ledger_reports_no_rows_rather_than_dividing_by_zero(self):
        out = ec.fee_scenarios(self.repo, B, 7.5)
        self.assertEqual(out["closed_trades"], 0)
        self.assertEqual(out["entry_turnover"], 0.0)
        for r in out["rows"]:
            self.assertIsNone(r["fee_bps_of_turnover"])


# ================================================== 5. ATR economics
class TestAtrEconomics(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.a, self.x = self.cfg.aggressive, self.cfg.execution

    def econ(self, atrs=(0.25, 1.00)):
        return ec.atr_economics(self.a, self.x, atrs)

    def rows(self, out):
        return {r["atr_pct"]: {t["tier"]: t for t in r["tiers"]}
                for r in out["rows"]}

    def test_stop_and_target_follow_the_config(self):
        r = self.econ((1.0,))["rows"][0]
        self.assertAlmostEqual(r["stop_pct"], self.a.stop_atr_mult, places=9)
        self.assertAlmostEqual(r["target_pct"],
                               self.a.target_r * self.a.stop_atr_mult, places=9)

    def test_zero_cost_breakeven_is_one_over_one_plus_r(self):
        # At 2R that is 33.3%. Every cost pushes it up from there, so a hit
        # rate below it cannot be rescued by any fee schedule.
        out = self.econ()
        self.assertAlmostEqual(out["zero_cost_breakeven_win_rate_pct"],
                               100.0 / 3.0, places=6)

    def test_the_round_trip_fee_is_twice_the_per_side_rate(self):
        t = self.rows(self.econ((1.0,)))[1.0]["tier1"]
        self.assertAlmostEqual(t["fee_round_trip_pct"], 2 * 80.0 / 100.0,
                               places=9)

    def test_a_winner_and_a_loser_pay_different_slippage(self):
        # A target exit is quote-priced; a stop is not. Collapsing them into
        # one number would understate the loss side on every trade.
        t = self.rows(self.econ((1.0,)))[1.0]["custom"]
        self.assertAlmostEqual(t["slippage_win_pct"],
                               2 * self.x.slippage_bps / 100.0, places=9)
        self.assertAlmostEqual(
            t["slippage_loss_pct"],
            (self.x.slippage_bps + self.x.stop_slippage_bps) / 100.0, places=9)
        self.assertGreater(t["slippage_loss_pct"], t["slippage_win_pct"])

    def test_untradeable_means_a_perfect_trade_still_loses(self):
        # NOT "the break-even win rate is high". At 0.25% ATR the 2R target is
        # 0.90% and tier 1 costs 1.60% in fees alone.
        t = self.rows(self.econ((0.25,)))[0.25]["tier1"]
        self.assertTrue(t["structurally_untradeable"])
        self.assertLess(t["net_target_pct"], 0.0)
        self.assertIsNone(t["breakeven_win_rate_pct"])

    def test_expensive_but_tradable_is_not_flagged_untradeable(self):
        t = self.rows(self.econ((0.25,)))[0.25]["tier3"]
        self.assertFalse(t["structurally_untradeable"])
        self.assertGreater(t["net_target_pct"], 0.0)
        self.assertGreater(t["breakeven_win_rate_pct"], 90.0)

    def test_breakeven_rises_with_the_fee_and_falls_with_atr(self):
        out = self.rows(self.econ((0.55, 1.50)))
        self.assertGreater(out[0.55]["tier6"]["breakeven_win_rate_pct"],
                           out[0.55]["custom"]["breakeven_win_rate_pct"])
        self.assertGreater(out[0.55]["custom"]["breakeven_win_rate_pct"],
                           out[1.50]["custom"]["breakeven_win_rate_pct"])

    def test_breakeven_matches_the_standalone_formula(self):
        t = self.rows(self.econ((0.70,)))[0.70]["tier4"]
        r = self.econ((0.70,))["rows"][0]
        want = ec.breakeven_win_rate(
            r["target_pct"], r["stop_pct"],
            t["fee_round_trip_pct"] + t["slippage_win_pct"],
            t["fee_round_trip_pct"] + t["slippage_loss_pct"])
        self.assertAlmostEqual(t["breakeven_win_rate_pct"], want, places=9)

    def test_the_minimum_tradable_atr_is_where_the_target_pays_for_a_winner(self):
        out = self.econ((0.25,))
        for name, floor in out["min_tradable_atr_pct"].items():
            t = self.rows(ec.atr_economics(self.a, self.x, (floor,)))[floor][name]
            # Exactly at the floor a perfect trade nets zero.
            self.assertAlmostEqual(t["net_target_pct"], 0.0, places=9)
            above = self.rows(ec.atr_economics(
                self.a, self.x, (floor + 0.01,)))[floor + 0.01][name]
            self.assertGreater(above["net_target_pct"], 0.0)

    def test_two_tiers_sit_above_the_configured_atr_floor(self):
        # The finding this table exists to produce: at the top two fee tiers,
        # setups the ATR filter admits cannot pay for themselves at all.
        out = self.econ()
        floors = out["min_tradable_atr_pct"]
        worse = [n for n, a in floors.items() if a > self.a.min_atr_pct]
        self.assertIn("tier1", worse)
        self.assertIn("tier2", worse)
        self.assertNotIn(FEE_TIER_CUSTOM, worse)

    def test_it_needs_no_trades_at_all(self):
        # The one part of this analysis that is not sample-limited.
        out = ec.atr_economics(self.a, self.x)
        self.assertEqual(len(out["rows"]), 6)
        self.assertTrue(all(math.isfinite(r["target_pct"]) for r in out["rows"]))

    def test_the_caveat_about_nine_exits_is_carried_in_the_data(self):
        self.assertIn("nine exits", self.econ()["caveat"])


# ================================ 6. the simulated fee must be VISIBLE
class TestTierIsReported(unittest.TestCase):
    """A paper result is only as honest as the fee it charged.

    A tier that changes fills without changing what the reports say is worse
    than no tier mechanism at all: it makes the operator confident about a
    number they never saw.
    """

    def test_the_preflight_summary_names_the_simulated_fee(self):
        from crypto_edge.verify_live import VerifyReport
        cfg = Config()
        cfg.execution.fee_tier = "tier2"
        rep = VerifyReport()
        out = rep.render_preflight(cfg)
        self.assertIn("FEE MODEL (SIMULATED)", out)
        self.assertIn("tier2", out)
        self.assertIn("60", out)

    def test_the_preflight_summary_names_a_custom_rate_too(self):
        cfg = Config()
        cfg.execution.taker_fee_bps = 7.5
        from crypto_edge.verify_live import VerifyReport
        out = VerifyReport().render_preflight(cfg)
        self.assertIn("custom", out)
        self.assertIn("7.5", out)


# ====================== 7. the mechanism must change nothing by default
class TestDefaultIsUnchanged(unittest.TestCase):
    def test_a_config_that_names_no_tier_prices_exactly_as_before(self):
        # The whole point of defaulting to "custom": adding the tier table is
        # not allowed to move a single existing fill.
        x = Config().execution
        self.assertEqual(x.effective_taker_bps(), 7.5)
        b = PaperBroker(x.effective_taker_bps(), x.slippage_bps,
                        x.stop_slippage_bps)
        self.assertAlmostEqual(b.fee(10_000.0), 7.5, places=9)

    def test_the_tier_table_is_data_not_behaviour(self):
        # Nothing in the fill path may consult the table directly; it reaches
        # the broker only through the config accessor.
        import inspect

        from crypto_edge.execution import paper_broker
        src = inspect.getsource(paper_broker)
        self.assertNotIn("KRAKEN_SPOT_TAKER_BPS", src)
        self.assertNotIn("fee_tier", src)


if __name__ == "__main__":
    unittest.main()
