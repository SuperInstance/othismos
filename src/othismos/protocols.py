"""
Protocol classes and type definitions for óthismos.

Provides formal typing for constraint functions, callbacks, and loggers.
Enables users to implement custom constraints/loggers with confidence
that they match the expected interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Sequence

import numpy as np


@runtime_checkable
class FeasibilityFn(Protocol):
    """Function that checks if a point is inside the feasible set."""

    def __call__(self, theta: np.ndarray) -> bool: ...


@runtime_checkable
class ProjectionFn(Protocol):
    """Function that projects a point onto the feasible set."""

    def __call__(self, theta: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class NormalFn(Protocol):
    """Function that returns the outward normal at a boundary point."""

    def __call__(self, theta: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class DistanceFn(Protocol):
    """Function measuring distance between two outputs (for context pressure)."""

    def __call__(self, a: object, b: object) -> float: ...


@runtime_checkable
class MetricLogger(Protocol):
    """Protocol for metric loggers (W&B, TensorBoard, DictLogger, etc.)."""

    def log(self, metrics: dict[str, float], step: int) -> None: ...


@runtime_checkable
class ConstraintLike(Protocol):
    """Full constraint protocol — what a Constraint must implement."""

    name: str
    type: object

    def is_feasible(self, theta: np.ndarray) -> bool: ...
    def project(self, theta: np.ndarray) -> np.ndarray: ...
