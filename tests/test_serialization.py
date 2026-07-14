"""Tests for óthismos serialization."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from othismos.pressure import PressureMeasurement, PressureGauge, l2_constraint, compute_othismos
from othismos.serialization import (
    save_history,
    load_history,
    export_metrics_csv,
    pressure_summary,
)
from othismos.diagnostics import PopcornDiagnostic, SystemHealth
from othismos.serialization import save_diagnostic


class TestSaveLoadHistory:
    def test_round_trip(self):
        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        theta = np.array([0.1, 0.0])

        for i in range(20):
            gradient = np.array([-float(i + 1) * 0.05, 0.0])
            gauge.measure(theta, gradient, 0.1, [c])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.json"
            save_history(gauge, path)
            assert path.exists()

            loaded = load_history(path)
            assert len(loaded.history) == 20
            assert loaded.current_pressure == pytest.approx(gauge.current_pressure)

    def test_json_structure(self):
        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        m = gauge.measure(np.array([0.1, 0.0]), np.array([-1.0, 0.0]), 0.1, [c])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hist.json"
            save_history(gauge, path)
            data = json.loads(path.read_text())
            assert "version" in data
            assert "measurements" in data
            assert data["measurements"][0]["step"] == 0


class TestCSVExport:
    def test_csv_export(self):
        gauge = PressureGauge()
        c = l2_constraint("l2", radius=0.1)
        for i in range(10):
            gauge.measure(np.array([0.1, 0.0]), np.array([-1.0, 0.0]), 0.1, [c])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.csv"
            export_metrics_csv(gauge, path)
            content = path.read_text()
            assert "step,pressure" in content
            assert "l2" in content
            assert len(content.strip().split("\n")) == 11  # header + 10 rows


class TestPressureSummary:
    def test_empty_gauge(self):
        gauge = PressureGauge()
        summary = pressure_summary(gauge)
        assert summary["status"] == "empty"

    def test_active_gauge(self):
        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        for i in range(20):
            gauge.measure(np.array([0.1, 0.0]), np.array([-1.0, 0.0]), 0.1, [c])

        summary = pressure_summary(gauge)
        assert summary["status"] == "active"
        assert summary["step_count"] == 20
        assert "goldilocks_zone" in summary
        assert "current_pressure" in summary
        assert "constraint_profile" in summary


class TestDiagnosticSerialization:
    def test_save_diagnostic(self):
        diag = PopcornDiagnostic()
        result = diag.diagnose([0.3, 0.4, 0.5, 0.6, 0.7], heat=1.0)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "diag.json"
            save_diagnostic(result, path)
            data = json.loads(path.read_text())
            assert data["health"] == result.health.value
            assert "recommendation" in data
