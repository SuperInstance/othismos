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

    def test_numpy_scalar_serialization(self):
        """Test that numpy scalar types (int64, float64) are correctly serialized."""
        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        # Use numpy scalars for step and pressure
        theta = np.array([0.1, 0.0])
        gradient = np.array([-1.0, 0.0])
        m = gauge.measure(theta, gradient, 0.1, [c])
        # Manually replace with numpy scalars to test serialization
        m.step = np.int64(42)
        m.pressure = np.float64(0.123456)
        m.pressure_by_constraint["test"] = np.float64(0.0001)
        # Also test numpy array with integer dtype
        m.desired_step = np.array([1, 2], dtype=np.int64)
        m.actual_step = np.array([3, 4], dtype=np.int32)
        m.violation = np.array([5, 6], dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history_numpy.json"
            # This should not raise a TypeError
            save_history(gauge, path)
            assert path.exists()

            # Load back and check that values are correct
            loaded = load_history(path)
            assert len(loaded.history) == 1
            loaded_m = loaded.history[0]
            # Check that numpy scalars became Python types
            assert isinstance(loaded_m.step, int)
            assert loaded_m.step == 42
            assert isinstance(loaded_m.pressure, float)
            assert loaded_m.pressure == pytest.approx(0.123456)
            assert isinstance(loaded_m.pressure_by_constraint["test"], float)
            assert loaded_m.pressure_by_constraint["test"] == pytest.approx(0.0001)
            # Arrays should be np.ndarray with values preserved
            assert isinstance(loaded_m.desired_step, np.ndarray)
            np.testing.assert_array_equal(loaded_m.desired_step, [1, 2])
            assert isinstance(loaded_m.actual_step, np.ndarray)
            np.testing.assert_array_equal(loaded_m.actual_step, [3, 4])
            assert isinstance(loaded_m.violation, np.ndarray)
            np.testing.assert_array_equal(loaded_m.violation, [5, 6])

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
