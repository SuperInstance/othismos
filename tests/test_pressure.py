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

    def test_l2_constraint_default_center_broadcasting(self):
        # Test that default center (origin) works for various theta shapes
        c = l2_constraint("weight_decay", radius=1.0)
        
        # 1-D theta
        theta1 = np.array([0.0, 0.0])  # origin
        assert c.is_feasible(theta1)
        theta1_out = np.array([2.0, 0.0])  # distance 2.0 > radius
        assert not c.is_feasible(theta1_out)
        projected = c.project(theta1_out)
        expected = np.array([1.0, 0.0])  # projected onto circle radius 1
        np.testing.assert_allclose(projected, expected, atol=1e-10)
        
        # 2-D theta (different length)
        theta2 = np.array([1.0, 1.0, 1.0])  # norm sqrt(3) ~ 1.732
        assert not c.is_feasible(theta2)  # outside radius 1
        projected2 = c.project(theta2)
        expected2 = np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)])  # unit vector
        np.testing.assert_allclose(projected2, expected2, atol=1e-10)
        
        # Verify that the constraint is indeed centered at origin
        # Point at [1,0,0] should be feasible (distance 1)
        theta3 = np.array([1.0, 0.0, 0.0])
        assert c.is_feasible(theta3)
        # Point at [1.1, 0, 0] should not be feasible
        theta3_out = np.array([1.1, 0.0, 0.0])
        assert not c.is_feasible(theta3_out)

    def test_l2_constraint_n_dimensional_theta(self):
        """Regression for Bug #1: l2_constraint must handle N-dimensional theta.

        Original bug: default center=np.zeros(1) broadcast incorrectly for
        theta with shape (n, d) where d > 1, causing wrong projection math.
        Fix: use np.zeros_like(theta) so center matches theta's shape.
        """
        c = l2_constraint("l2", radius=5.0)

        # Test 1-D theta of various lengths
        for shape in [(2,), (3,), (10,), (100,)]:
            theta = np.ones(shape)  # shape-dependent norm
            # feasible if norm <= radius
            assert c.is_feasible(theta) == (np.linalg.norm(theta) <= 5.0 + 1e-10)
            projected = c.project(theta)
            if np.linalg.norm(theta) > 5.0:
                assert abs(np.linalg.norm(projected) - 5.0) < 1e-9

        # Test 2-D theta
        theta_2d = np.array([[3.0, 4.0], [1.0, 1.0]])  # shape (2, 2), norm sqrt(27)
        assert not c.is_feasible(theta_2d)
        projected_2d = c.project(theta_2d)
        assert projected_2d.shape == theta_2d.shape
        assert abs(np.linalg.norm(projected_2d) - 5.0) < 1e-9

        # Test 3-D theta
        theta_3d = np.ones((3, 4, 5))  # 60 elements, each 1.0 → norm sqrt(60)
        assert not c.is_feasible(theta_3d)
        projected_3d = c.project(theta_3d)
        assert projected_3d.shape == theta_3d.shape

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

    def test_store_vectors_memory_efficiency(self):
        """Regression for Bug #6: PressureGauge should support memory-efficient mode.

        For large models (1B+ params), storing full numpy arrays per step
        causes OOM. With store_vectors=False, only scalar norms are kept,
        reducing per-step memory from O(n_params) to O(1).
        """
        c = l2_constraint("l2", radius=0.5)
        # Simulate large model: 10M params
        n_params = 10_000_000
        theta = np.zeros(n_params, dtype=np.float32)
        gradient = np.ones(n_params, dtype=np.float32) * 1e-6

        # Memory-efficient mode
        gauge_mem = PressureGauge(store_vectors=False, window_size=100)
        m = gauge_mem.measure(theta, gradient, 0.01, [c])

        # Vectors should be zero-size (not the full 10M floats)
        assert m.desired_step.size == 0
        assert m.actual_step.size == 0
        assert m.violation.size == 0
        # But scalars should be intact
        assert isinstance(m.pressure, float)
        assert isinstance(m.desired_norm, float)
        assert isinstance(m.violation_norm, float)
        # scalar_dict() should be usable for memory-efficient logging
        summary = m.scalar_dict()
        assert "pressure" in summary
        assert "desired_norm" in summary
        assert summary["pressure"] >= 0

        # Many measurements shouldn't blow memory
        for _ in range(50):
            gauge_mem.measure(theta, gradient, 0.01, [c])
        assert len(gauge_mem.history) == 51

    def test_store_vectors_flag(self):
        """Test that store_vectors=False zeros out vector arrays but keeps scalars."""
        c = l2_constraint("test", radius=0.5)
        gauge = PressureGauge(store_vectors=False)
        theta = np.array([0.3, 0.4])
        gradient = np.array([-1.0, 0.0])

        m = gauge.measure(theta, gradient, 0.1, [c])

        # Scalars should be present and be Python floats
        assert isinstance(m.pressure, float)
        assert isinstance(m.desired_norm, float)
        assert isinstance(m.actual_norm, float)
        assert isinstance(m.violation_norm, float)
        assert m.pressure >= 0
        assert m.desired_norm >= 0
        assert m.actual_norm >= 0
        assert m.violation_norm >= 0

        # Vector arrays should be empty (zero length) when store_vectors=False
        assert m.desired_step.size == 0
        assert m.actual_step.size == 0
        assert m.violation.size == 0
        # They should be np.ndarray with shape (0,)
        assert isinstance(m.desired_step, np.ndarray)
        assert isinstance(m.actual_step, np.ndarray)
        assert isinstance(m.violation, np.ndarray)
        assert m.desired_step.shape == (0,)
        assert m.actual_step.shape == (0,)
        assert m.violation.shape == (0,)

        # Now test with store_vectors=True (default)
        gauge2 = PressureGauge(store_vectors=True)
        m2 = gauge2.measure(theta, gradient, 0.1, [c])
        assert m2.desired_step.size > 0
        assert m2.actual_step.size > 0
        assert m2.violation.size > 0
        # Check that they have the correct shape (should match theta)
        assert m2.desired_step.shape == theta.shape
        assert m2.actual_step.shape == theta.shape
        assert m2.violation.shape == theta.shape
        # Scalars should also be present
        assert isinstance(m2.pressure, float)
        assert isinstance(m2.desired_norm, float)
        assert isinstance(m2.actual_norm, float)
        assert isinstance(m2.violation_norm, float)
