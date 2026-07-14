"""
Pressure measurement — the core óthismos computation.

Implements Π (instantaneous pressure) and aggregate metrics
from math/01_PRESSURE_MATH.md.

The constraint violation attempt Δθ = s − s* measures how hard
a system is pushing against its constraints. Π = ‖Δθ‖.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class ConstraintType(str, Enum):
    """Types of constraints that produce óthismos."""

    L2_NORM = "l2_norm"          # Weight regularization bounds
    CONTEXT_WINDOW = "context"   # Token/context limits
    THERMAL = "thermal"          # Thermal/energy budget
    COMPUTE = "compute"          # FLOPS/cycle budget
    MEMORY = "memory"            # RAM/storage limits
    LATENCY = "latency"          # Time deadlines
    CUSTOM = "custom"            # User-defined constraint


@dataclass
class Constraint:
    """A single hard constraint on a system.

    The feasible set C is defined by: constraint(value) returns True
    if value is inside C, False otherwise. The projection P_C(value)
    returns the nearest point in C.
    """

    name: str
    type: ConstraintType
    bound_fn: callable
    project_fn: callable
    normal_fn: callable | None = None  # outward normal at boundary, if known

    def is_feasible(self, theta: np.ndarray) -> bool:
        return self.bound_fn(theta)

    def project(self, theta: np.ndarray) -> np.ndarray:
        return self.project_fn(theta)


@dataclass
class PressureMeasurement:
    """A single óthismos measurement at optimization step t.

    Attributes:
        step: Optimization step index
        desired_step: s(t) — the unconstrained gradient step
        actual_step: s*(t) — the projected (constrained) step
        violation: Δθ = s − s*, the constraint violation attempt
        pressure: Π = ‖Δθ‖, instantaneous óthismos
        pressure_by_constraint: Per-constraint pressure breakdown
    """

    step: int
    desired_step: np.ndarray
    actual_step: np.ndarray
    violation: np.ndarray
    pressure: float
    pressure_by_constraint: dict[str, float] = field(default_factory=dict)

    @property
    def is_pushing(self) -> bool:
        """True if the system is exerting measurable pressure."""
        return self.pressure > 1e-10


def _l2_ball_project(theta: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Project theta onto an L2 ball of given radius around center."""
    diff = theta - center
    norm = np.linalg.norm(diff)
    if norm <= radius:
        return theta
    return center + diff * (radius / norm)


def l2_constraint(name: str, radius: float, center: np.ndarray | None = None) -> Constraint:
    """Create an L2-norm constraint (e.g., weight regularization)."""
    center = center if center is not None else np.zeros(1)
    return Constraint(
        name=name,
        type=ConstraintType.L2_NORM,
        bound_fn=lambda theta: np.linalg.norm(theta - center) <= radius,
        project_fn=lambda theta: _l2_ball_project(theta, center, radius),
        normal_fn=lambda theta: (theta - center) / max(np.linalg.norm(theta - center), 1e-12),
    )


def box_constraint(name: str, lows: np.ndarray, highs: np.ndarray) -> Constraint:
    """Create a box constraint (element-wise bounds)."""
    def box_project(theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, lows, highs)

    return Constraint(
        name=name,
        type=ConstraintType.CUSTOM,
        bound_fn=lambda theta: bool(np.all(theta >= lows) and np.all(theta <= highs)),
        project_fn=box_project,
    )


def compute_othismos(
    theta: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    constraints: Sequence[Constraint],
) -> PressureMeasurement:
    """Compute instantaneous óthismos at a single optimization step.

    Args:
        theta: Current parameter vector θ(t)
        gradient: Loss gradient ∇ℒ(θ(t))
        learning_rate: Step size η
        constraints: List of active constraints

    Returns:
        PressureMeasurement with full breakdown
    """
    # The step the system WANTS to take
    desired = -learning_rate * gradient

    # Apply constraints sequentially (projection onto intersection)
    theta_new = theta + desired
    pressure_by: dict[str, float] = {}

    for constraint in constraints:
        projected = constraint.project(theta_new)
        violation_component = theta_new - projected
        individual_pressure = float(np.linalg.norm(violation_component))
        if individual_pressure > 1e-12:
            pressure_by[constraint.name] = individual_pressure
        theta_new = projected

    # Total violation
    actual = theta_new - theta
    violation = desired - actual
    total_pressure = float(np.linalg.norm(violation))

    return PressureMeasurement(
        step=-1,  # caller can override
        desired_step=desired,
        actual_step=actual,
        violation=violation,
        pressure=total_pressure,
        pressure_by_constraint=pressure_by,
    )


@dataclass
class GoldilocksZone:
    """The productive pressure range for a system.

    Below lower_bound: the system is in Burn or Dormancy (not enough
        pressure to produce information).
    Above upper_bound: the system is in Crisis/Rupture risk (too much
        pressure for structural integrity).
    Within bounds: the system is in Resistance — the most productive phase.
    """

    lower_bound: float
    upper_bound: float

    def contains(self, pressure: float) -> bool:
        return self.lower_bound <= pressure <= self.upper_bound

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound


def goldilocks_range(
    measurements: Sequence[PressureMeasurement],
    percentile_low: float = 25.0,
    percentile_high: float = 90.0,
) -> GoldilocksZone:
    """Estimate the Goldilocks pressure zone from historical data.

    The lower bound is where the system starts producing useful work
    (above the 25th percentile of observed pressure).
    The upper bound is where structural failure risk increases
    (90th percentile, before the tail of crisis events).
    """
    if not measurements:
        return GoldilocksZone(lower_bound=0.0, upper_bound=1.0)

    pressures = [m.pressure for m in measurements]
    low = float(np.percentile(pressures, percentile_low))
    high = float(np.percentile(pressures, percentile_high))
    return GoldilocksZone(lower_bound=low, upper_bound=high)


class PressureGauge:
    """A running óthismos gauge for a system.

    Accumulates pressure measurements over time and provides
    aggregate statistics, trend detection, and alerting.
    """

    def __init__(self, window_size: int = 1000):
        self._history: list[PressureMeasurement] = []
        self._window = window_size
        self._step = 0

    def measure(
        self,
        theta: np.ndarray,
        gradient: np.ndarray,
        learning_rate: float,
        constraints: Sequence[Constraint],
    ) -> PressureMeasurement:
        """Take a pressure reading."""
        m = compute_othismos(theta, gradient, learning_rate, constraints)
        m.step = self._step
        self._step += 1
        self._history.append(m)
        if len(self._history) > self._window:
            self._history = self._history[-self._window:]
        return m

    @property
    def current_pressure(self) -> float:
        return self._history[-1].pressure if self._history else 0.0

    @property
    def mean_pressure(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean([m.pressure for m in self._history]))

    @property
    def pressure_trend(self) -> float:
        """Slope of pressure over recent window. Positive = rising."""
        if len(self._history) < 2:
            return 0.0
        n = min(len(self._history), 50)
        recent = [m.pressure for m in self._history[-n:]]
        x = np.arange(n)
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    def goldilocks(self) -> GoldilocksZone:
        return goldilocks_range(self._history)

    @property
    def history(self) -> list[PressureMeasurement]:
        return list(self._history)

    def pressure_profile(self) -> dict[str, float]:
        """Per-constraint pressure breakdown, averaged over history."""
        if not self._history:
            return {}
        accumulators: dict[str, list[float]] = {}
        for m in self._history:
            for name, val in m.pressure_by_constraint.items():
                accumulators.setdefault(name, []).append(val)
        return {
            name: float(np.mean(vals)) for name, vals in accumulators.items()
        }
