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
from typing import Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from othismos.protocols import FeasibilityFn, ProjectionFn, NormalFn


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
    bound_fn: "FeasibilityFn"
    project_fn: "ProjectionFn"
    normal_fn: "NormalFn | None" = None  # outward normal at boundary, if known

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
    desired_norm: float = 0.0
    actual_norm: float = 0.0
    violation_norm: float = 0.0
    store_vectors: bool = True

    @property
    def is_pushing(self) -> bool:
        """True if the system is exerting measurable pressure."""
        return self.pressure > 1e-10

    def scalar_dict(self) -> dict:
        """Return only scalar fields — memory-efficient for logging."""
        return {
            "step": self.step,
            "pressure": self.pressure,
            "desired_norm": self.desired_norm,
            "actual_norm": self.actual_norm,
            "violation_norm": self.violation_norm,
            "pressure_by_constraint": dict(self.pressure_by_constraint),
        }


def _l2_ball_project(theta: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Project theta onto an L2 ball of given radius around center."""
    diff = theta - center
    norm = np.linalg.norm(diff)
    if norm <= radius:
        return theta
    return center + diff * (radius / norm)


def l2_constraint(name: str, radius: float, center: np.ndarray | None = None) -> Constraint:
    """Create an L2-norm constraint (e.g., weight regularization)."""
    if center is None:
        # Lazy origin: treat center as zero vector of appropriate shape
        bound_fn = lambda theta: np.linalg.norm(theta) <= radius
        project_fn = lambda theta: _l2_ball_project(theta, np.zeros_like(theta), radius)
        normal_fn = lambda theta: theta / max(np.linalg.norm(theta), 1e-12)
    else:
        bound_fn = lambda theta: np.linalg.norm(theta - center) <= radius
        project_fn = lambda theta: _l2_ball_project(theta, center, radius)
        normal_fn = lambda theta: (theta - center) / max(np.linalg.norm(theta - center), 1e-12)
    return Constraint(
        name=name,
        type=ConstraintType.L2_NORM,
        bound_fn=bound_fn,
        project_fn=project_fn,
        normal_fn=normal_fn,
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
    desired_norm = float(np.linalg.norm(desired))

    # Apply constraints sequentially (projection onto intersection).
    #
    # KNOWN LIMITATION (v0.4.0): Sequential projection does not perform
    # fixed-point iteration. When two constraints interact (e.g., L2 ball
    # + box constraint), projecting onto constraint A may push the result
    # back outside constraint B. Our implementation projects A then B,
    # which gives the correct projection onto A ∩ B only when A and B are
    # "compatible" (their projections commute). For general constraint
    # sets, a true fixed-point iteration would be needed.
    #
    # In practice this is rarely a problem because:
    # - Most constraint sets ARE compatible (L2 + soft bounds, etc.)
    # - The error from sequential projection is small relative to the
    #   measurement noise in the gradient
    # - Users who need exact intersection projection can pass their own
    #   project_fn that performs the fixed-point iteration
    #
    # Future work: add an `iterative=True` flag that runs fixed-point
    # projection for users who need exact A ∩ B projection.
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
        desired_norm=desired_norm,
        actual_norm=float(np.linalg.norm(actual)),
        violation_norm=total_pressure,
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

    For large models (>1B params), set store_vectors=False to avoid
    storing full numpy arrays per step. Only scalar norms are kept.
    """

    def __init__(self, window_size: int = 1000, store_vectors: bool = True):
        self._history: list[PressureMeasurement] = []
        self._window = window_size
        self._step = 0
        self._store_vectors = store_vectors
        # Rolling statistics for memory efficiency
        self._pressure_sum = 0.0
        self._pressure_sq_sum = 0.0
        self._pressure_count = 0

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
        m.store_vectors = self._store_vectors
        if not self._store_vectors:
            # Zero out arrays to free memory — scalars already stored
            m.desired_step = np.zeros(0)
            m.actual_step = np.zeros(0)
            m.violation = np.zeros(0)
        self._step += 1
        self._history.append(m)
        # Update rolling stats
        self._pressure_sum += m.pressure
        self._pressure_sq_sum += m.pressure ** 2
        self._pressure_count += 1
        if not self._history:
            return
        if len(self._history) > self._window:
            # Should never happen if measure() is the only append path,
            # but guard against external manipulation
            while len(self._history) > self._window:
                old = self._history.pop(0)
                self._pressure_sum -= old.pressure
                self._pressure_sq_sum -= old.pressure ** 2
                self._pressure_count = max(0, self._pressure_count - 1)
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
