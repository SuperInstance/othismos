"""Tests for projection-free context pressure."""

import numpy as np
import pytest

from othismos.context_pressure import (
    ContextPressureGauge,
    ContextPressureMeasurement,
    cosine_distance,
    l2_distance,
    token_overlap,
)


class TestDistanceFunctions:
    def test_l2_distance(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        d = l2_distance(a, b)
        assert d == pytest.approx(np.sqrt(2))

    def test_l2_distance_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert l2_distance(a, a) == pytest.approx(0.0)

    def test_cosine_distance(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(0.0)

        c = np.array([0.0, 1.0])
        assert cosine_distance(a, c) == pytest.approx(1.0)

    def test_token_overlap_identical(self):
        tokens = ["hello", "world", "foo"]
        assert token_overlap(tokens, tokens) == pytest.approx(0.0)

    def test_token_overlap_disjoint(self):
        a = ["hello", "world"]
        b = ["foo", "bar"]
        assert token_overlap(a, b) == pytest.approx(1.0)

    def test_token_overlap_partial(self):
        a = ["hello", "world", "foo"]
        b = ["hello", "bar", "baz"]
        # intersection: hello (1), union: hello,world,foo,bar,baz (5)
        assert token_overlap(a, b) == pytest.approx(1.0 - 1.0 / 5.0)


class TestContextPressureGauge:
    def test_default_kl_distance(self):
        gauge = ContextPressureGauge()
        p1 = np.array([0.5, 0.5])
        p2 = np.array([0.5, 0.5])
        m = gauge.measure(p1, p2, context_tokens_dropped=0)
        assert m.pressure == pytest.approx(0.0, abs=1e-6)

    def test_kl_distance_different(self):
        gauge = ContextPressureGauge()
        p1 = np.array([0.9, 0.1])
        p2 = np.array([0.1, 0.9])
        m = gauge.measure(p1, p2, context_tokens_dropped=100)
        assert m.pressure > 0

    def test_custom_distance_fn(self):
        gauge = ContextPressureGauge(distance_fn=l2_distance)
        m = gauge.measure(np.array([1.0]), np.array([3.0]))
        assert m.pressure == pytest.approx(2.0)

    def test_history_tracking(self):
        gauge = ContextPressureGauge()
        for i in range(10):
            gauge.measure(
                np.array([1.0, 0.0]),
                np.array([1.0 - i * 0.1, i * 0.1]),
                context_tokens_dropped=i * 10,
            )
        assert len(gauge.history) == 10
        assert gauge.current_pressure > 0

    def test_mean_and_trend(self):
        gauge = ContextPressureGauge()
        for i in range(20):
            gauge.measure(
                np.array([0.5, 0.5]),
                np.array([0.5 + i * 0.02, 0.5 - i * 0.02]),
            )
        assert gauge.mean_pressure > 0
        assert gauge.pressure_trend > 0  # increasing divergence

    def test_pressure_vs_dropped(self):
        gauge = ContextPressureGauge()
        gauge.measure(np.array([1.0, 0.0]), np.array([0.5, 0.5]), context_tokens_dropped=50)
        gauge.measure(np.array([1.0, 0.0]), np.array([0.3, 0.7]), context_tokens_dropped=100)

        pairs = gauge.pressure_vs_dropped()
        assert len(pairs) == 2
        assert pairs[0] == (50, pytest.approx(pairs[0][1]))

    def test_window_size(self):
        gauge = ContextPressureGauge(window_size=5)
        for i in range(10):
            gauge.measure(np.array([1.0]), np.array([0.0]))
        assert len(gauge.history) == 5

    def test_metadata_storage(self):
        gauge = ContextPressureGauge()
        m = gauge.measure(
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
            context_tokens_dropped=100,
            model="gpt-test",
            prompt_length=500,
        )
        assert m.metadata["model"] == "gpt-test"
        assert m.metadata["prompt_length"] == 500
