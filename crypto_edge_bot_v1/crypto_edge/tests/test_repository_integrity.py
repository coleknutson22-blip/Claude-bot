"""The repository must actually contain the code it claims to.

DEFECT UNDER TEST
-----------------
`.gitignore` carried an unanchored `data/` rule, intended for the runtime
SQLite folder at the project root. Git patterns without a leading slash match a
directory of that name at ANY depth, so it also matched the Python package
`crypto_edge/data/`. Files already tracked stayed tracked -- `.gitignore` does
not untrack anything -- so nothing looked wrong locally, and every test passed
on the development machine because the file was present on disk.

But `crypto_edge/data/broad_universe.py` was NEW, so `git add -A` silently
skipped it, and the pushed branch contained an `engine.py` that imports a
module the repository does not ship. A fresh clone died with
ModuleNotFoundError before a single test ran.

The lesson is not "remember to check". A test suite that passes only on the
machine that wrote it is not evidence of anything, so the check is automated
here: every module the package contains must be tracked by git, and no source
file may be matched by an ignore rule.
"""
import subprocess
import unittest
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "crypto_edge"


def git(*args, cwd=PROJECT_ROOT):
    """Run a git command, or return None when git/the repo is unavailable."""
    try:
        out = subprocess.run(("git",) + args, cwd=str(cwd), capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def source_files():
    """Every Python source file that makes up the shipped package."""
    return sorted(p for p in PACKAGE_DIR.rglob("*.py")
                  if "__pycache__" not in p.parts)


class TestEverySourceFileIsCommitted(unittest.TestCase):
    def setUp(self):
        if git("rev-parse", "--git-dir") is None:
            self.skipTest("not a git checkout (released tarball or no git)")

    def test_every_package_module_is_tracked_by_git(self):
        """The exact failure: a module on disk that the repository does not have."""
        listed = git("ls-files", "--", str(PACKAGE_DIR))
        self.assertIsNotNone(listed, "git ls-files failed")
        tracked = {Path(line).name for line in listed.splitlines() if line.strip()}

        missing = [p.relative_to(PROJECT_ROOT).as_posix()
                   for p in source_files() if p.name not in tracked]
        self.assertEqual(
            missing, [],
            "these modules exist on disk but are NOT in the repository, so a "
            "fresh clone cannot import them:\n  " + "\n  ".join(missing))

    def test_no_source_file_is_matched_by_an_ignore_rule(self):
        """Catches the cause rather than the symptom: an over-broad pattern."""
        paths = [str(p) for p in source_files()]
        proc = subprocess.run(["git", "check-ignore", "-v", *paths],
                              cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        # exit 1 means "nothing ignored", which is what we want
        offenders = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        self.assertEqual(
            offenders, [],
            "these source files are excluded by a .gitignore rule:\n  "
            + "\n  ".join(offenders))

    def test_test_files_are_tracked_too(self):
        listed = git("ls-files", "--", str(PROJECT_ROOT / "tests"))
        self.assertIsNotNone(listed)
        tracked = {Path(line).name for line in listed.splitlines() if line.strip()}
        on_disk = {p.name for p in (PROJECT_ROOT / "tests").glob("*.py")}
        self.assertEqual(sorted(on_disk - tracked), [],
                         "untracked test files would not reach a fresh clone")

    def test_the_runtime_data_folder_is_still_ignored(self):
        """The original intent must survive the fix: runtime state stays out."""
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "data/crypto_edge.db"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         "the runtime database must still be gitignored")

    def test_the_env_file_is_still_ignored_but_the_example_is_not(self):
        secret = subprocess.run(["git", "check-ignore", "-q", ".env"],
                                cwd=str(PROJECT_ROOT), capture_output=True)
        example = subprocess.run(["git", "check-ignore", "-q", ".env.example"],
                                 cwd=str(PROJECT_ROOT), capture_output=True)
        self.assertEqual(secret.returncode, 0, ".env must stay ignored")
        self.assertNotEqual(example.returncode, 0,
                            ".env.example must be committed")


class TestEveryModuleImports(unittest.TestCase):
    """A cheap, environment-independent guard: the package must import."""

    def test_every_module_in_the_package_imports_cleanly(self):
        import importlib

        failures = []
        for path in source_files():
            rel = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
            name = ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                importlib.import_module(name)
            except Exception as e:            # noqa: BLE001 -- report them all
                failures.append(f"{name}: {type(e).__name__}: {e}")
        self.assertEqual(failures, [],
                         "modules that do not import:\n  " + "\n  ".join(failures))

    def test_the_module_that_went_missing_is_importable_and_complete(self):
        from crypto_edge.data import broad_universe

        for attr in ("BroadUniverseService", "CoinGeckoUniverseProvider",
                     "StaticBroadUniverseProvider", "RankedAsset",
                     "BroadUniverse", "content_hash", "NON_ASSET_BASES"):
            self.assertTrue(hasattr(broad_universe, attr),
                            f"broad_universe.{attr} is missing")

    def test_the_engine_can_import_what_it_declares(self):
        """engine.py is what failed on the operator's machine."""
        from crypto_edge import engine

        self.assertTrue(hasattr(engine, "TradingEngine"))
        self.assertTrue(hasattr(engine, "_build_broad_provider"))


if __name__ == "__main__":
    unittest.main()
