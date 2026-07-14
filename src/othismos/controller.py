"""
Pressure-driven adaptive control.

Implements the PressureController from the applications research:
automatically adjusts learning rate, constraint tightness, and other
knobs based on the current molt phase and diagnostic health.

The controller is the "automatic transmission" for óthismos:
- Expansion → speed up (increase LR)
- Resistance → hold steady
- Crisis → reduce LR, relax constraints (let the system molt)
- Settlement → hold steady, let new envelope solidify
- Dormancy → increase LR or inject new problem

Usage:
    >>> from othismos import PressureController
    >>> controller = PressureController(gauge=my_gauge)
    >>> # In training loop:
    >>> actions = controller.update(current_lr, current_constraints)
    >>> for action in actions:
    ...     action.apply(optimizer, model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import numpy as np

from othismos.pressure import PressureGauge, GoldilocksZone, Constraint
from othismos.phases import MoltPhase, MoltCycleTracker
from othismos.diagnostics import PopcornDiagnostic, SystemHealth


class ActionType(str, Enum):
    """Types of control actions the controller can take."""

    ADJUST_LR = "adjust_lr"           # Multiply LR by a factor
    RELAX_CONSTRAINT = "relax"        # Loosen a constraint
    TIGHTEN_CONSTRAINT = "tighten"    # Tighten a constraint
    ALERT = "alert"                    # Surface a message to the operator
    CHECKPOINT = "checkpoint"          # Save current state (pre-molt)


@dataclass
class ControlAction:
    """A single action recommended by the controller."""

    type: ActionType
    factor: float = 1.0
    constraint_name: str | None = None
    message: str = ""
    priority: int = 0  # higher = more urgent

    def __repr__(self) -> str:
        if self.type == ActionType.ADJUST_LR:
            return f"AdjustLR(×{self.factor:.2f})"
        elif self.type == ActionType.RELAX_CONSTRAINT:
            return f"RelaxConstraint('{self.constraint_name}', ×{self.factor:.2f})"
        elif self.type == ActionType.TIGHTEN_CONSTRAINT:
            return f"TightenConstraint('{self.constraint_name}', ×{self.factor:.2f})"
        elif self.type == ActionType.ALERT:
            return f"Alert('{self.message}')"
        elif self.type == ActionType.CHECKPOINT:
            return f"Checkpoint('{self.message}')"
        return f"Action({self.type})"


class PressureController:
    """Adaptive controller driven by óthismos measurements.

    Monitors pressure, classifies the molt phase, runs diagnostics,
    and recommends control actions.

    Args:
        gauge: A PressureGauge with history
        lr_bounds: (min_lr, max_lr) to clamp learning rate adjustments
        crisis_lr_factor: Multiply LR by this when in Crisis (default: 0.5)
        expansion_lr_factor: Multiply LR by this when in Expansion (default: 1.5)
        burn_patience: Steps to wait in Burn before alerting (default: 50)
    """

    def __init__(
        self,
        gauge: PressureGauge,
        tracker: MoltCycleTracker | None = None,
        lr_bounds: tuple[float, float] = (1e-6, 1.0),
        crisis_lr_factor: float = 0.5,
        expansion_lr_factor: float = 1.3,
        burn_patience: int = 50,
    ) -> None:
        self.gauge = gauge
        self.tracker = tracker or MoltCycleTracker()
        self.lr_bounds = lr_bounds
        self.crisis_lr_factor = crisis_lr_factor
        self.expansion_lr_factor = expansion_lr_factor
        self.burn_patience = burn_patience

        self._diag = PopcornDiagnostic()
        self._burn_counter = 0
        self._seep_counter = 0
        self._last_phase: MoltPhase | None = None
        self._step = 0

    def update(
        self,
        current_lr: float,
        constraints: list[Constraint] | None = None,
        heat: float | None = None,
    ) -> list[ControlAction]:
        """Check current state and return recommended actions.

        Call this once per training step. Apply returned actions
        to your optimizer/constraints as needed.

        Args:
            current_lr: Current learning rate
            constraints: Current constraint set (for relaxation recommendations)
            heat: External heat (if None, estimated from gradient norms)

        Returns:
            List of ControlActions, sorted by priority (highest first).
        """
        actions: list[ControlAction] = []
        self._step += 1

        if not self.gauge.history:
            return actions

        pressures = [m.pressure for m in self.gauge.history]

        # Estimate heat if not provided
        if heat is None:
            heat = max(pressures) if pressures else 1.0

        # Phase classification
        reading = self.tracker.update(pressures[-1])

        # Diagnostic
        diag = self._diag.diagnose(pressures, heat=heat)

        new_lr = current_lr
        changed_lr = False

        # ─── Phase-based LR adjustment ────────────────────────────

        if reading.phase == MoltPhase.CRISIS:
            factor = self.crisis_lr_factor
            new_lr = max(self.lr_bounds[0], current_lr * factor)
            if new_lr != current_lr:
                actions.append(ControlAction(
                    type=ActionType.ADJUST_LR,
                    factor=factor,
                    message=f"Crisis phase: reducing LR ({current_lr:.6f} → {new_lr:.6f})",
                    priority=10,
                ))
                changed_lr = True

            # Suggest checkpoint before potential molt
            if self._last_phase != MoltPhase.CRISIS:
                actions.append(ControlAction(
                    type=ActionType.CHECKPOINT,
                    message="Entering Crisis — checkpoint before molt",
                    priority=9,
                ))

        elif reading.phase == MoltPhase.EXPANSION and self._last_phase in (
            MoltPhase.SETTLEMENT, MoltPhase.DORMANCY, None
        ):
            factor = self.expansion_lr_factor
            new_lr = min(self.lr_bounds[1], current_lr * factor)
            if new_lr != current_lr:
                actions.append(ControlAction(
                    type=ActionType.ADJUST_LR,
                    factor=factor,
                    message=f"Expansion phase: increasing LR ({current_lr:.6f} → {new_lr:.6f})",
                    priority=5,
                ))
                changed_lr = True

        # ─── Diagnostic-based actions ─────────────────────────────

        if diag.health == SystemHealth.BURN:
            self._burn_counter += 1
            self._seep_counter = 0

            if self._burn_counter >= self.burn_patience:
                actions.append(ControlAction(
                    type=ActionType.ALERT,
                    message=(
                        f"BURN detected for {self._burn_counter} steps. "
                        "The system has no internal pressure. "
                        "Consider: harder problem, different data, new constraint shape."
                    ),
                    priority=8,
                ))
                # Suggest tightening constraints to create pressure
                if constraints:
                    actions.append(ControlAction(
                        type=ActionType.TIGHTEN_CONSTRAINT,
                        constraint_name=constraints[0].name,
                        factor=0.5,
                        message="Tighten constraints to create productive pressure",
                        priority=7,
                    ))

        elif diag.health == SystemHealth.SEEP:
            self._seep_counter += 1
            self._burn_counter = 0

            if self._seep_counter >= self.burn_patience:
                actions.append(ControlAction(
                    type=ActionType.ALERT,
                    message=(
                        f"SEEP detected for {self._seep_counter} steps. "
                        "Pressure is leaking. Check for: too-weak regularization, "
                        "unstable data pipeline, gradient noise."
                    ),
                    priority=6,
                ))

        else:
            self._burn_counter = max(0, self._burn_counter - 1)
            self._seep_counter = max(0, self._seep_counter - 1)

        # ─── Relax constraints in sustained Crisis ──────────────────

        if (
            reading.phase == MoltPhase.CRISIS
            and constraints
            and len(self.gauge.history) > 20
        ):
            zone = self.gauge.goldilocks()
            current_pressure = self.gauge.current_pressure

            if current_pressure > zone.upper_bound * 1.5:
                # Pressure way above Goldilocks — relax the binding constraint
                binding = max(
                    self.gauge.pressure_profile().items(),
                    key=lambda kv: kv[1],
                    default=(None, 0.0),
                )
                if binding[0]:
                    actions.append(ControlAction(
                        type=ActionType.RELAX_CONSTRAINT,
                        constraint_name=binding[0],
                        factor=1.3,
                        message=f"Relaxing '{binding[0]}' — pressure {current_pressure:.4f} >> {zone.upper_bound:.4f}",
                        priority=7,
                    ))

        self._last_phase = reading.phase

        # Sort by priority
        actions.sort(key=lambda a: -a.priority)
        return actions

    def status(self) -> dict:
        """Return current controller status for logging."""
        reading = None
        if self.tracker.current_phase:
            reading = self.tracker.current_phase

        return {
            "step": self._step,
            "phase": reading.label if reading else "Unknown",
            "current_pressure": self.gauge.current_pressure,
            "mean_pressure": self.gauge.mean_pressure,
            "burn_counter": self._burn_counter,
            "seep_counter": self._seep_counter,
            "cycles": self.tracker.cycle_count,
        }
