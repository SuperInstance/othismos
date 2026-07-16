"""
Integrations for óthismos with ML frameworks.

Provides callback classes for PyTorch, HuggingFace Trainer, and
generic metric logging (W&B, TensorBoard, dict-based).

Usage with PyTorch:
    >>> from othismos.integrations import OthismosTorchCallback
    >>> callback = OthismosTorchCallback(constraints=[...])
    >>> # In training loop:
    >>> callback.pre_step(model, optimizer, loss)
    >>> optimizer.step()
    >>> callback.post_step(model)

Usage with HuggingFace Trainer:
    >>> from transformers import Trainer
    >>> from othismos.integrations import OthismosTrainerCallback
    >>> trainer = Trainer(
    ...     callbacks=[OthismosTrainerCallback(constraints=[...])],
    ...     ...
    ... )
"""

from __future__ import annotations

from typing import Any, Sequence, Protocol

import numpy as np

from othismos.pressure import (
    Constraint,
    PressureGauge,
    GoldilocksZone,
    compute_othismos,
    goldilocks_range,
)
from othismos.phases import MoltCycleTracker, MoltPhase
from othismos.diagnostics import PopcornDiagnostic, SystemHealth


class MetricLogger(Protocol):
    """Protocol for metric loggers (W&B, TensorBoard, etc.)."""

    def log(self, metrics: dict[str, float], step: int) -> None: ...


class DictLogger:
    """Simple dict-based logger for testing or custom pipelines."""

    def __init__(self) -> None:
        self._data: list[tuple[int, dict[str, float]]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._data.append((step, dict(metrics)))

    @property
    def history(self) -> list[tuple[int, dict[str, float]]]:
        return list(self._data)

    def metric_series(self, name: str) -> list[float]:
        return [m.get(name, float("nan")) for _, m in self._data]


class OthismosTorchCallback:
    """PyTorch training callback for óthismos monitoring.

    Attaches to a PyTorch training loop and measures constraint pressure
    at each optimizer step. Tracks molt cycles and runs the popcorn
    diagnostic automatically.

    The constraint set is applied to model parameters after optimizer.step()
    via projection. Pressure is measured as the gap between the optimizer's
    desired step and the projected step. Optionally, the constrained parameters
    can be written back to the model.

    Args:
        constraints: List of othismos Constraint objects (applied to flattened params)
        logger: Optional MetricLogger (W&B, TensorBox, DictLogger)
        log_every: Log metrics every N steps (default: every step)
        auto_clip_grad: If True, clip gradients to the Goldilocks zone upper bound
        apply_constraints: If True, write constrained parameters back to model
                           after measuring pressure (default: False, observatory-only)
    """

    def __init__(
        self,
        constraints: Sequence[Constraint],
        logger: MetricLogger | None = None,
        log_every: int = 1,
        auto_clip_grad: bool = False,
        apply_constraints: bool = False,
    ) -> None:
        self.constraints = list(constraints)
        self.logger = logger
        self.log_every = max(1, log_every)
        self.auto_clip_grad = auto_clip_grad
        self.apply_constraints = apply_constraints

        self.gauge = PressureGauge(window_size=10000)
        self.tracker = MoltCycleTracker()
        self.diagnostic = PopcornDiagnostic()

        self._step = 0
        self._global_step = 0
        self._pre_step_params: np.ndarray | None = None

    def _flatten_params(self, model) -> np.ndarray:
        """Extract flattened parameter vector from a PyTorch model."""
        import torch
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

    def _flatten_grads(self, model) -> np.ndarray:
        """Extract flattened gradient vector from a PyTorch model."""
        import torch
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().cpu().numpy().ravel())
            else:
                grads.append(np.zeros(p.numel()))
        return np.concatenate(grads)

    def _write_back_params(self, model, flat: np.ndarray) -> None:
        """Write flattened params back into model."""
        import torch
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data = torch.from_numpy(flat[idx:idx + n].reshape(p.shape)).to(p.device, dtype=p.dtype)
            idx += n

    def pre_step(self, model, optimizer, loss) -> None:
        """Call BEFORE optimizer.step(). Captures pre-step parameter state."""
        self._pre_step_params = self._flatten_params(model)

    def post_step(self, model) -> dict[str, float]:
        """Call AFTER optimizer.step(). Measures pressure, updates tracking.

        Returns a metrics dict suitable for logging.
        """
        if self._pre_step_params is None:
            return {}

        post_params = self._flatten_params(model)

        # The actual step taken
        actual_step = post_params - self._pre_step_params

        # Compute what the unconstrained step would have been
        # (reconstruct from optimizer state if available, else approximate)
        # For standard SGD/Adam, the unconstrained step is the pre-step + grad direction
        # We measure pressure by applying our constraints to see what gets clipped
        projected = self._pre_step_params.copy()
        for c in self.constraints:
            projected = c.project(self._pre_step_params + actual_step)

        # Pressure = how much our constraints would clip the step
        violation = (self._pre_step_params + actual_step) - projected
        pressure = float(np.linalg.norm(violation))

        # Update gauge
        from othismos.pressure import PressureMeasurement
        m = PressureMeasurement(
            step=self._step,
            desired_step=actual_step,
            actual_step=projected - self._pre_step_params,
            violation=violation,
            pressure=pressure,
        )
        self.gauge._history.append(m)
        self.gauge._step += 1
        self._step += 1

        # Optionally write back constrained parameters to model
        if self.apply_constraints:
            self._write_back_params(model, projected)

        # Phase tracking
        reading = self.tracker.update(pressure)

        # Build metrics
        metrics: dict[str, float] = {
            "othismos/pressure": pressure,
            "othismos/mean_pressure": self.gauge.mean_pressure,
            "othismos/pressure_trend": self.gauge.pressure_trend,
            "othismos/phase": int(reading.phase),
            "othismos/phase_confidence": reading.confidence,
            "othismos/cycle_count": float(self.tracker.cycle_count),
        }

        # Goldilocks zone (after enough data)
        if len(self.gauge.history) >= 10:
            zone = self.gauge.goldilocks()
            metrics["othismos/goldilocks_lower"] = zone.lower_bound
            metrics["othismos/goldilocks_upper"] = zone.upper_bound

        # Per-constraint breakdown
        profile = self.gauge.pressure_profile()
        for name, val in profile.items():
            metrics[f"othismos/constraint_{name}"] = val

        # Auto gradient clipping based on Goldilocks zone
        if self.auto_clip_grad and len(self.gauge.history) >= 20:
            zone = self.gauge.goldilocks()
            if pressure > zone.upper_bound:
                # System under too much pressure — clip gradients to reduce
                clip_factor = zone.upper_bound / max(pressure, 1e-12)
                import torch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_factor)
                metrics["othismos/grad_clip_factor"] = clip_factor

        # Log
        if self.logger and (self._global_step % self.log_every == 0):
            self.logger.log(metrics, self._global_step)

        self._global_step += 1
        return metrics

    def health_report(self) -> str:
        """Generate a human-readable health report."""
        if not self.gauge.history:
            return "No measurements yet."

        all_pressures = [m.pressure for m in self.gauge.history]
        heat = max(all_pressures)  # proxy
        diag = self.diagnostic.diagnose(all_pressures, heat=heat)

        phase = self.tracker.current_phase
        phase_name = phase.label if phase else "Unknown"
        zone = self.gauge.goldilocks()

        lines = [
            "═" * 60,
            "Óthismos Health Report",
            "═" * 60,
            f"  Health:           {diag.health.value.upper()}",
            f"  Current phase:    {phase_name}",
            f"  Current Π:        {self.gauge.current_pressure:.6f}",
            f"  Mean Π:           {self.gauge.mean_pressure:.6f}",
            f"  Pressure trend:   {self.gauge.pressure_trend:+.6f}",
            f"  Goldilocks zone:  [{zone.lower_bound:.6f}, {zone.upper_bound:.6f}]",
            f"  Cycles detected:  {self.tracker.cycle_count}",
            f"  Confidence:       {diag.confidence:.2f}",
            "",
            f"  Recommendation: {diag.recommendation}",
            "═" * 60,
        ]
        return "\n".join(lines)


class OthismosTrainerCallback:
    """HuggingFace Trainer callback wrapper.

    Wraps OthismosTorchCallback for HF Trainer's callback API.

    Usage:
        >>> from transformers import Trainer
        >>> cb = OthismosTrainerCallback(constraints=[...])
        >>> trainer = Trainer(callbacks=[cb], ...)
    """

    def __init__(
        self,
        constraints: Sequence[Constraint],
        logger: MetricLogger | None = None,
        log_every: int = 1,
    ) -> None:
        self._inner = OthismosTorchCallback(constraints, logger, log_every)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._inner.pre_step(model, None, None)

    def on_post_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is not None:
            metrics = self._inner.post_step(model)
            # HF logs metrics from the returned dict if we set them on state
            # Actually, HF callback API expects us to log manually
            if metrics and self._inner.logger:
                self._inner.logger.log(metrics, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Print final health report
        report = self._inner.health_report()
        print(report)


def constraint_from_torch_model(
    model,
    max_norm: float | None = None,
    max_per_param: float | None = None,
) -> list[Constraint]:
    """Build constraints from a PyTorch model's properties.

    Args:
        model: PyTorch model (used only for parameter shapes if needed)
        max_norm: If set, creates an L2 constraint with this radius
        max_per_param: If set, creates box constraints per parameter

    Returns:
        List of Constraint objects
    """
    constraints: list[Constraint] = []

    if max_norm is not None:
        constraints.append(
            Constraint(
                name="global_l2",
                type=None,  # type: ignore
                bound_fn=lambda theta: np.linalg.norm(theta) <= max_norm,
                project_fn=lambda theta: (
                    theta if np.linalg.norm(theta) <= max_norm
                    else theta * (max_norm / np.linalg.norm(theta))
                ),
            )
        )

    if max_per_param is not None:
        n_params = sum(p.numel() for p in model.parameters())
        constraints.append(
            Constraint(
                name="per_param_clip",
                type=None,  # type: ignore
                bound_fn=lambda theta: bool(np.all(np.abs(theta) <= max_per_param)),
                project_fn=lambda theta: np.clip(theta, -max_per_param, max_per_param),
            )
        )

    return constraints
