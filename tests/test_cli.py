"""Tests for the óthismos CLI."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from othismos.cli import main as cli_main


def run_cli(args: list[str]) -> tuple[str, str]:
    """Run the CLI with given args, capture output."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    old_argv = sys.argv
    sys.argv = ["othismos"] + args
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            cli_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    return out_buf.getvalue(), err_buf.getvalue()


class TestCLIVersion:
    def test_version(self):
        out, _ = run_cli(["version"])
        assert "othismos" in out
        assert "0.3" in out


class TestCLIReef:
    def test_reef_add_and_stats(self, tmp_path):
        db = str(tmp_path / "reef.json")
        out, _ = run_cli(["reef", "add", "d1", "First deposit", "--db", db])
        assert "ACCEPTED" in out or "✓" in out

        out, _ = run_cli(["reef", "add", "d2", "Second deposit", "--refs", "d1", "--db", db])
        assert "ACCEPTED" in out or "✓" in out

        out, _ = run_cli(["reef", "stats", "--db", db])
        assert "total_deposits" in out

    def test_reef_list(self, tmp_path):
        db = str(tmp_path / "reef.json")
        run_cli(["reef", "add", "base", "Foundation deposit", "--db", db])
        run_cli(["reef", "add", "child", "Child deposit", "--refs", "base", "--db", db])
        out, _ = run_cli(["reef", "list", "--db", db])
        assert "base" in out
        assert "child" in out

    def test_reef_search(self, tmp_path):
        db = str(tmp_path / "reef.json")
        run_cli(["reef", "add", "d1", "machine learning optimization", "--db", db])
        run_cli(["reef", "add", "d2", "quantum computing", "--db", db])
        out, _ = run_cli(["reef", "search", "--query", "learning", "--db", db])
        assert "d1" in out
        assert "quantum" not in out

    def test_reef_tick(self, tmp_path):
        db = str(tmp_path / "reef.json")
        run_cli(["reef", "add", "d1", "test", "--db", db])
        out, _ = run_cli(["reef", "tick", "--db", db])
        assert "Step 1" in out

    def test_reef_fail(self, tmp_path):
        db = str(tmp_path / "reef.json")
        run_cli(["reef", "add", "foundation", "Critical", "--db", db])
        run_cli(["reef", "add", "child", "Depends on foundation", "--refs", "foundation", "--db", db])
        out, _ = run_cli(["reef", "fail", "foundation", "--db", db])
        assert "REEFQUAKE" in out

    def test_reef_graph(self, tmp_path):
        db = str(tmp_path / "reef.json")
        run_cli(["reef", "add", "a", "Root", "--db", db])
        run_cli(["reef", "add", "b", "Child", "--refs", "a", "--db", db])
        out, _ = run_cli(["reef", "graph", "--db", db])
        data = json.loads(out)
        assert "b" in data
        assert data["b"] == ["a"]


class TestCLIDiagnose:
    def test_diagnose(self, tmp_path):
        from othismos.pressure import PressureGauge, l2_constraint
        from othismos.serialization import save_history
        import numpy as np

        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        for i in range(20):
            gauge.measure(
                np.array([0.1, 0.0]),
                np.array([-float(i + 1) * 0.05, 0.0]),
                0.1,
                [c],
            )

        hist_path = str(tmp_path / "hist.json")
        save_history(gauge, hist_path)

        out, _ = run_cli(["diagnose", hist_path, "--heat", "1.0"])
        assert "Health:" in out
