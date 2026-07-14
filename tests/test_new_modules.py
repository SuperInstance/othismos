"""Tests for protocols, config, viz, and pandas export."""

import numpy as np
import pytest

from othismos.protocols import (
    FeasibilityFn,
    ProjectionFn,
    MetricLogger,
    ConstraintLike,
)
from othismos.config import OthismosConfig
from othismos.pressure import (
    PressureGauge,
    PressureMeasurement,
    l2_constraint,
    box_constraint,
)
from othismos.phases import MoltCycleTracker, PhaseClassifier


class TestProtocols:
    def test_constraint_satisfies_protocol(self):
        c = l2_constraint("test", radius=1.0)
        # Constraint has name, type, is_feasible, project
        assert hasattr(c, "name")
        assert hasattr(c, "is_feasible")
        assert hasattr(c, "project")

    def test_custom_feasibility_fn(self):
        def my_feasibility(theta: np.ndarray) -> bool:
            return bool(np.all(np.abs(theta) < 1))
        assert callable(my_feasibility)
        assert isinstance(my_feasibility, type(my_feasibility))  # it's a function


class TestConfig:
    def test_defaults(self):
        config = OthismosConfig()
        assert config.window_size == 1000
        assert config.burn_threshold == 0.01
        assert config.lr_bounds == (1e-6, 1.0)

    def test_from_dict(self):
        config = OthismosConfig.from_dict({"crisis_threshold": 0.5, "window_size": 500})
        assert config.crisis_threshold == 0.5
        assert config.window_size == 500

    def test_from_dict_ignores_unknown(self):
        config = OthismosConfig.from_dict({"crisis_threshold": 0.5, "unknown_key": "ignored"})
        assert config.crisis_threshold == 0.5

    def test_to_dict(self):
        config = OthismosConfig(crisis_threshold=0.3)
        d = config.to_dict()
        assert d["crisis_threshold"] == 0.3
        assert "window_size" in d

    def test_build_gauge(self):
        config = OthismosConfig(window_size=200)
        gauge = config.build_gauge()
        assert gauge._window == 200

    def test_build_classifier(self):
        config = OthismosConfig(crisis_threshold=0.5, expansion_floor=0.1)
        clf = config.build_classifier()
        assert clf.crisis_threshold == 0.5
        assert clf.expansion_floor == 0.1

    def test_build_diagnostic(self):
        config = OthismosConfig(burn_threshold=0.05)
        diag = config.build_diagnostic()
        assert diag.burn_threshold == 0.05

    def test_build_controller(self):
        config = OthismosConfig(crisis_lr_factor=0.7)
        controller = config.build_controller()
        assert controller.crisis_lr_factor == 0.7

    def test_build_all(self):
        config = OthismosConfig()
        gauge, tracker, diagnostic, controller = config.build_all()
        assert gauge is not None
        assert tracker is not None
        assert diagnostic is not None
        assert controller is not None

    def test_yaml_roundtrip(self, tmp_path):
        config = OthismosConfig(crisis_threshold=0.42, window_size=999)
        yaml_path = str(tmp_path / "config.yaml")
        config.to_yaml(yaml_path)

        loaded = OthismosConfig.from_yaml(yaml_path)
        assert loaded.crisis_threshold == 0.42
        assert loaded.window_size == 999


class TestViz:
    def test_plot_pressure_empty(self):
        from othismos.viz import plot_pressure
        gauge = PressureGauge()
        fig = plot_pressure(gauge)
        assert fig is not None

    def test_plot_pressure_with_data(self):
        from othismos.viz import plot_pressure
        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        for i in range(20):
            gauge.measure(
                np.array([0.1, 0.0]),
                np.array([-float(i + 1) * 0.05, 0.0]),
                0.1,
                [c],
            )
        fig = plot_pressure(gauge, show_goldilocks=True, show_trend=True)
        assert fig is not None

    def test_plot_constraint_profile(self):
        from othismos.viz import plot_constraint_profile
        gauge = PressureGauge()
        c1 = l2_constraint("l2", radius=0.1)
        c2 = box_constraint("box", lows=np.array([-0.05, -0.05]), highs=np.array([0.05, 0.05]))
        for i in range(15):
            gauge.measure(
                np.array([0.08, 0.08]),
                np.array([-0.5, -0.5]),
                0.5,
                [c1, c2],
            )
        fig = plot_constraint_profile(gauge)
        assert fig is not None

    def test_plot_molt_cycle_empty(self):
        from othismos.viz import plot_molt_cycle
        tracker = MoltCycleTracker()
        fig = plot_molt_cycle(tracker)
        assert fig is not None

    def test_plot_diagnostic_timeline(self):
        from othismos.viz import plot_diagnostic_timeline
        gauge = PressureGauge()
        c = l2_constraint("test", radius=0.1)
        for i in range(30):
            gauge.measure(
                np.array([0.1, 0.0]),
                np.array([-float(i + 1) * 0.02, 0.0]),
                0.1,
                [c],
            )
        fig = plot_diagnostic_timeline(gauge, heat=0.5, window=10)
        assert fig is not None


class TestPandasExport:
    def test_gauge_to_dataframe_empty(self):
        from othismos.pandas_export import gauge_to_dataframe
        gauge = PressureGauge()
        df = gauge_to_dataframe(gauge)
        assert df.empty

    def test_gauge_to_dataframe(self):
        from othismos.pandas_export import gauge_to_dataframe
        gauge = PressureGauge()
        c = l2_constraint("l2", radius=0.1)
        for i in range(20):
            gauge.measure(
                np.array([0.1, 0.0]),
                np.array([-0.5, 0.0]),
                0.1,
                [c],
            )
        df = gauge_to_dataframe(gauge)
        assert len(df) == 20
        assert "step" in df.columns
        assert "pressure" in df.columns
        assert "constraint_l2" in df.columns

    def test_tracker_to_dataframe(self):
        from othismos.pandas_export import tracker_to_dataframe
        from othismos.phases import PhaseClassifier
        classifier = PhaseClassifier(crisis_threshold=0.5, expansion_floor=0.1)
        tracker = MoltCycleTracker(classifier=classifier)
        for p in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.2, 0.1, 0.05, 0.15, 0.25]:
            tracker.update(p)
        df = tracker_to_dataframe(tracker)
        assert len(df) > 0
        assert "phase" in df.columns
        assert "pressure" in df.columns

    def test_reef_to_dataframe(self):
        from othismos.pandas_export import reef_to_dataframe
        from othismos.ecology import Reef
        reef = Reef()
        reef.submit("a", "Foundation deposit")
        reef.submit("b", "Child of a", references=["a"])
        reef.submit("c", "Child of b", references=["b"])
        df = reef_to_dataframe(reef)
        assert len(df) == 3
        assert "id" in df.columns
        assert "layer" in df.columns
        assert "depth_score" in df.columns
