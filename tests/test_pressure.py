"""Tests for the óthismos pressure measurement module."""

import math
import numpy as np
import pytest

from othismos.pressure import (
    ConstraintType,
    Constraint,
    PressureMeasurement,
    PressureGauge,
    GoldilocksZone,
    compute_othismos,
    goldilocks_range,
    l2_constraint,
    box_constraint,
)


class TestConstraints:
    def test_l2_constraint_inside(self):
        c = l2_constraint("weight_decay", radius=1.0)
        theta = np.array([0.3, 0.4])  # norm = 0.5 < 1.0
        assert c.is_feasible(theta)
        projected = c.project(theta)
        np.testing.assert_allclose(projected, theta)

    def test_l2_constraint_outside(self):
        c = l2_constraint("weight_decay", radius=1.0)
        theta = np.array([0.6, 0.8])  # norm = 1.0 — on boundary
        assert c.is_feasible(theta)
        theta_out = np.array([1.2, 1.6])  # norm = 2.0 > 1.0
        assert not c.is_feasible(theta_out)
        projected = c.project(theta_out)
        np.testing.assert_allclose(projected, [0.6, 0.8], atol=1e-10)

    def test_l2_constraint_with_center(self):
        center = np.array([1.0, 1.0])
        c = l2_constraint("bounded", radius=0.5, center=center)
        theta = np.array([2.0, 2.0])  # distance = sqrt(2) > 0.5
        projected = c.project(theta)
        diff = projected - center
        assert abs(np.linalg.norm(diff) - 0.5) < 1e-10

    def test_box_constraint(self):
        c = box_constraint("clip", lows=np.array([-1.0, -1.0]), highs=np.array([1.0, 1.0]))
        theta = np.array([0.5, 0.5])
        assert c.is_feasible(theta)
        theta_out = np.array([2.0, -3.0])
        assert not c.is_feasible(theta_out)
        projected = c.project(theta_out)
        np.testing.assert_allclose(projected, [1.0, -1.0])


class TestComputeOthismos:
    def test_no_constraints_no_pressure(self):
        theta = np.array([1.0, 1.0])
        gradient = np.array([0.5, 0.5])
        m = compute_othismos(theta, gradient, 0.1, constraints=[])
        assert m.pressure < 1e-10
        assert not m.is_pushing

    def test_constraint_produces_pressure(self):
        c = l2_constraint("weight_decay", radius=0.1)
        # theta at boundary, gradient pointing outward
        theta = np.array([0.1, 0.0])
        gradient = np.array([-1.0, 0.0])  # step would be -0.1 * (-1, 0) = (0.1, 0) → outward
        m = compute_othismos(theta, gradient, 0.1, constraints=[c])
        assert m.pressure > 0
        assert m.is_pushing
        assert "weight_decay" in m.pressure_by_constraint

    def test_gradient_pointing_inward_no_pressure(self):
        c = l2_constraint("weight_decay", radius=1.0)
        theta = np.array([0.5, 0.0])  # inside
        gradient = np.array([1.0, 0.0])  # step = -0.1 * (1,0) = (-0.1, 0) → inward
        m = compute_othismos(theta, gradient, 0.1, constraints=[c])
        assert m.pressure < 1e-10

    def test_multiple_constraints(self):
        c1 = l2_constraint("l2", radius=0.5)
        c2 = box_constraint("box", lows=np.array([-0.3, -0.3]), highs=np.array([0.3, 0.3]))
        theta = np.array([0.25, 0.25])
        gradient = np.array([-1.0, -1.0])  # strong outward push
        m = compute_othismos(theta, gradient, 0.5, constraints=[c1, c2])
        assert m.pressure > 0
        # Box constraint should be the binding one here
        assert "box" in m.pressure_by_constraint

    def test_measurement_properties(self):
        c = l2_constraint("test", radius=0.1)
        theta = np.array([0.1, 0.0])
        gradient = np.array([-1.0, 0.0])
        m = compute_othismos(theta, gradient, 0.1, constraints=[c])
        assert isinstance(m, PressureMeasurement)
        assert m.desired_step.shape == (2,)
        assert m.actual_step.shape == (2,)
        assert m.violation.shape == (2,)
        assert m.pressure >= 0


class TestGoldilocksZone:
    def test_contains(self):
        zone = GoldilocksZone(lower_bound=0.1, upper_bound=0.9)
        assert zone.contains(0.5)
        assert zone.contains(0.1)
        assert zone.contains(0.9)
        assert not zone.contains(0.05)
        assert not zone.contains(0.95)

    def test_width(self):
        zone = GoldilocksZone(lower_bound=0.2, upper_bound=0.8)
        assert math.isclose(zone.width, 0.6)


class TestGoldilocksRange:
    def test_empty_measurements(self):
        zone = goldilocks_range([])
        assert zone.lower_bound == 0.0

    def test_from_measurements(self):
        measurements = [
            PressureMeasurement(step=i, desired_step=np.zeros(2), actual_step=np.zeros(2),
                              violation=np.zeros(2), pressure=float(i) * 0.1)
            for i in range(20)
        ]
        zone = goldilocks_range(measurements)
        assert zone.lower_bound > 0
        assert zone.upper_bound > zone.lower_bound


class TestPressureGauge:
    def test_gauge_tracking(self):
        c = l2_constraint("test", radius=0.5)
        gauge = PressureGauge(window_size=100)
        theta = np.array([0.4, 0.0])

        for i in range(50):
            gradient = np.array([-float(i) * 0.01, 0.0])
            m = gauge.measure(theta, gradient, 0.1, [c])
            assert m.step == i

        assert len(gauge.history) == 50
        assert gauge.current_pressure >= 0
        assert gauge.mean_pressure >= 0

    def test_pressure_trend(self):
        c = l2_constraint("test", radius=0.1)
        gauge = PressureGauge()
        theta = np.array([0.1, 0.0])

        for i in range(20):
            gradient = np.array([-float(i + 1) * 0.05, 0.0])
            gauge.measure(theta, gradient, 0.1, [c])

        # Pressure should be trending upward (increasing gradient)
        assert gauge.pressure_trend > 0

    def test_pressure_profile(self):
        c1 = l2_constraint("l2", radius=0.3)
        c2 = box_constraint("box", lows=np.array([-0.2, -0.2]), highs=np.array([0.2, 0.2]))
        gauge = PressureGauge()
        theta = np.array([0.15, 0.15])

        for _ in range(10):
            gradient = np.array([-0.5, -0.5])
            gauge.measure(theta, gradient, 0.5, [c1, c2])

        profile = gauge.pressure_profile()
        assert "l2" in profile or "box" in profile
