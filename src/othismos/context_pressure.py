"""
Projection-free pressure measurement.

For systems where you can't define a projection operator — LLMs with
context windows, black-box APIs, human-in-the-loop systems.

Instead of measuring ‖s − s*‖ (which requires knowing s*), we measure
the behavioral difference between constrained and unconstrained execution.
This is the "projection-free" óthismos: pressure inferred from output
divergence rather than parameter-space violation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from othismos.pressure import PressureMeasurement


@dataclass
class ContextPressureMeasurement:
    """Pressure measurement for projection-free (behavioral) óthismos.

    Instead of measuring parameter-space violation, measures the KL
    divergence (or other distance) between the full-context output
    and the constrained-context output.

    Attributes:
        pressure: Divergence between unconstrained and constrained outputs
        context_tokens_dropped: How many tokens were removed
        full_output: The unconstrained output
        constrained_output: The constrained output
    """

    pressure: float
    context_tokens_dropped: int
    full_output: Any
    constrained_output: Any
    method: str = "kl_divergence"
    metadata: dict = field(default_factory=dict)


class ContextPressureGauge:
    """Measure óthismos for systems with context/sequence constraints.

    Useful for LLMs where the constraint is a context window limit.
    Measures how much the output changes when context is truncated.

    Usage:
        >>> gauge = ContextPressureGauge(distance_fn=my_distance_fn)
        >>> result = gauge.measure(
        ...     full_output=full_logits,
        ...     constrained_output=truncated_logits,
        ...     context_tokens_dropped=500,
        ... )
    """

    def __init__(
        self,
        distance_fn: Callable[[Any, Any], float] | None = None,
        window_size: int = 1000,
    ) -> None:
        """
        Args:
            distance_fn: Function(output_a, output_b) → float.
                Default: KL divergence on probability distributions.
            window_size: How many measurements to keep.
        """
        self._distance_fn = distance_fn or self._default_kl_distance
        self._history: list[ContextPressureMeasurement] = []
        self._window = window_size

    @staticmethod
    def _default_kl_distance(p_full: np.ndarray, p_constrained: np.ndarray) -> float:
        """Symmetric KL divergence between two probability distributions."""
        p_full = np.asarray(p_full, dtype=np.float64)
        p_constrained = np.asarray(p_constrained, dtype=np.float64)

        # Add small epsilon to avoid log(0)
        eps = 1e-12
        p_full = p_full + eps
        p_constrained = p_constrained + eps

        # Normalize
        p_full = p_full / p_full.sum()
        p_constrained = p_constrained / p_constrained.sum()

        # Symmetric KL (Jeffrey's divergence)
        kl_1 = np.sum(p_full * np.log(p_full / p_constrained))
        kl_2 = np.sum(p_constrained * np.log(p_constrained / p_full))
        return float(kl_1 + kl_2)

    def measure(
        self,
        full_output: Any,
        constrained_output: Any,
        context_tokens_dropped: int = 0,
        **metadata,
    ) -> ContextPressureMeasurement:
        """Measure pressure between full and constrained outputs."""
        pressure = self._distance_fn(full_output, constrained_output)

        m = ContextPressureMeasurement(
            pressure=pressure,
            context_tokens_dropped=context_tokens_dropped,
            full_output=full_output,
            constrained_output=constrained_output,
            metadata=metadata,
        )
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
        if len(self._history) < 2:
            return 0.0
        n = min(len(self._history), 50)
        recent = [m.pressure for m in self._history[-n:]]
        x = np.arange(n)
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    def pressure_vs_dropped(self) -> list[tuple[int, float]]:
        """Pairs of (tokens_dropped, pressure) for analyzing scaling."""
        return [(m.context_tokens_dropped, m.pressure) for m in self._history]

    @property
    def history(self) -> list[ContextPressureMeasurement]:
        return list(self._history)


def cosine_distance(a: Any, b: Any) -> float:
    """Cosine distance for embedding-based pressure measurement."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(1.0 - dot / norm)


def l2_distance(a: Any, b: Any) -> float:
    """L2 distance for logit/embedding pressure."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def token_overlap(full_tokens: Sequence[str], constrained_tokens: Sequence[str]) -> float:
    """Measure pressure as 1 - token overlap (Jaccard distance)."""
    full_set = set(full_tokens)
    constrained_set = set(constrained_tokens)
    if not full_set and not constrained_set:
        return 0.0
    intersection = len(full_set & constrained_set)
    union = len(full_set | constrained_set)
    return 1.0 - (intersection / union if union > 0 else 0.0)
