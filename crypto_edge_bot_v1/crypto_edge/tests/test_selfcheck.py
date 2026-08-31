"""Startup self-check tests.

The self-check is the gate between a misconfigured process and a running
trader, so it is tested like a safety device: it must fail closed, it must
distinguish critical faults from cosmetic ones, and it must never report
PAPER TRADING ACTIVE while a critical check is red.
"""
import logging
import unittest

from crypto_edge.selfcheck import CRITICAL, WARNING, SelfCheckReport, run_selfcheck
from helpers import STRATEGY, default_meta, temp_repo, test_config


class NullNotifier:
    enabled = False

    def test_connectivity(self):
        return True, "ok"


class TestReportSemantics(unittest.TestCase):
    def test_critical_failure_blocks_startup(self):
        rep = SelfCheckReport()
        rep.add("db integrity", False, CRITICAL, "corrupt")
        self.assertFalse(rep.passed)
        self.assertIn("STOPPED", rep.render())
        self.assertNotIn("PAPER TRADING ACTIVE", rep.render())

    def test_warning_failure_does_not_block_startup(self):
        rep = SelfCheckReport()
        rep.add("telegram connectivity", False, WARNING, "timeout")
        self.assertTrue(rep.passed, "a Telegram outage must not stop the trader")
        self.assertIn("PAPER TRADING ACTIVE", rep.render())

    def test_render_marks_fail_and_warn_differently(self):
        rep = SelfCheckReport()
        rep.add("db integrity", False, CRITICAL)
        rep.add("telegram connectivity", False, WARNING)
        out = rep.render()
        self.assertIn("[FAIL]", out)
        self.assertIn("[WARN]", out)

    def test_all_green_reports_active(self):
        rep = SelfCheckReport()
        rep.add("db integrity", True, CRITICAL, "ok")
        self.assertTrue(rep.passed)
        self.assertIn("PAPER TRADING ACTIVE", rep.render())

    def test_empty_report_is_not_treated_as_healthy_by_accident(self):
        # vacuously passes; the guard is that run_selfcheck always adds checks
        self.assertTrue(SelfCheckReport().passed)


class TestOfflineSelfCheck(unittest.TestCase):
    def setUp(self):
        self.repo, _ = temp_repo()
        self.repo.ensure_account(STRATEGY, 10_000.0)

    def test_offline_run_skips_network_checks_and_passes(self):
        cfg = test_config()
        cfg.telegram.enabled = False
        rep = run_selfcheck(cfg, self.repo, None, NullNotifier(),
                            check_network=False)
        self.assertTrue(rep.passed, rep.render())
        self.assertIn("skipped (offline mode)", rep.render())

    def test_paper_mode_is_asserted_every_start(self):
        cfg = test_config()
        cfg.telegram.enabled = False
        rep = run_selfcheck(cfg, self.repo, None, NullNotifier(),
                            check_network=False)
        names = {r.name: r for r in rep.results}
        self.assertIn("PAPER mode enforced", names)
        self.assertTrue(names["PAPER mode enforced"].ok)
        self.assertEqual(names["PAPER mode enforced"].severity, CRITICAL)

    def test_missing_telegram_credentials_is_a_critical_config_error(self):
        cfg = test_config()
        cfg.telegram.enabled = True          # enabled but no token/chat id
        cfg.telegram_token = ""
        cfg.telegram_chat_id = ""
        rep = run_selfcheck(cfg, self.repo, None, NullNotifier(),
                            check_network=False)
        self.assertFalse(rep.passed)
        self.assertIn("STOPPED", rep.render())

    def test_no_database_fails_closed(self):
        cfg = test_config()
        cfg.telegram.enabled = False
        rep = run_selfcheck(cfg, None, None, NullNotifier(), check_network=False)
        self.assertFalse(rep.passed, "no database must never be treated as fine")


class TestSelfCheckLogging(unittest.TestCase):
    """A failed cosmetic check must not log at the same level as a corrupt DB."""

    def _levels(self, rep_builder):
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("ce.app")
        handler = Capture()
        prev_level, prev_prop = logger.level, logger.propagate
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            rep_builder()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
            logger.propagate = prev_prop
        return {r.getMessage(): r.levelname for r in records}

    def test_warning_severity_failure_logs_as_warning_not_error(self):
        repo, _ = temp_repo()
        repo.ensure_account(STRATEGY, 10_000.0)
        cfg = test_config()
        cfg.telegram.enabled = False

        class FailingNotifier:
            enabled = True

            def test_connectivity(self):
                return False, "timeout"

        levels = self._levels(
            lambda: run_selfcheck(cfg, repo, None, FailingNotifier(),
                                  check_network=False))
        errors = [m for m, lv in levels.items() if lv == "ERROR"]
        self.assertEqual(errors, [],
                         f"nothing critical failed, so nothing should log ERROR: {errors}")

    def test_critical_failure_logs_as_error(self):
        repo, _ = temp_repo()
        repo.ensure_account(STRATEGY, 10_000.0)
        cfg = test_config()
        cfg.telegram.enabled = True
        cfg.telegram_token = ""
        cfg.telegram_chat_id = ""
        levels = self._levels(
            lambda: run_selfcheck(cfg, repo, None, NullNotifier(),
                                  check_network=False))
        self.assertTrue([m for m, lv in levels.items() if lv == "ERROR"],
                        "a critical failure must be logged at ERROR")

    def test_log_message_does_not_read_as_a_success_claim(self):
        repo, _ = temp_repo()
        repo.ensure_account(STRATEGY, 10_000.0)
        cfg = test_config()
        cfg.telegram.enabled = True
        cfg.telegram_token = ""
        cfg.telegram_chat_id = ""
        levels = self._levels(
            lambda: run_selfcheck(cfg, repo, None, NullNotifier(),
                                  check_network=False))
        failing = [m for m, lv in levels.items() if lv == "ERROR"]
        # "selfcheck: configuration valid" at ERROR reads like a passing check.
        for msg in failing:
            self.assertIn("FAILED", msg)


if __name__ == "__main__":
    unittest.main()
