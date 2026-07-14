# Óthismos Library Design — Engineering Research

> **Research Document 02**
> How to make óthismos useful as a Python library that real engineers reach for.

---

## Table of Contents

1. [Framework Integration](#1-framework-integration)
2. [API Audit](#2-api-audit)
3. [Packaging & Distribution](#3-packaging--distribution)
4. [Comparison with Similar Libraries](#4-comparison-with-similar-libraries)
5. [Recommended Patterns — Code Examples](#5-recommended-patterns--code-examples)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Framework Integration

Óthismos measures something no other library does: the *pressure* a system exerts against its constraints during optimization. To be useful, it must plug into the training/optimization loops people already use. Below is each major framework, what the integration looks like, and the callback API óthismos should expose.

### 1.1 The Core Callback Protocol

Óthismos needs a single unified callback interface that all framework integrations adapt to. This is the heart of library usability.

```python
# src/othismos/callbacks.py (PROPOSED)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence
import numpy as np

from othismos.pressure import (
    Constraint, PressureMeasurement, PressureGauge,
    compute_othismos,
)
from othismos.phases import MoltPhase, PhaseClassifier, PhaseReading, MoltCycleTracker
from othismos.diagnostics import PopcornDiagnostic, DiagnosticResult


@dataclass
class StepContext:
    """Everything óthismos needs to know about the current step.

    Framework adapters populate this; óthismos callbacks consume it.
    """
    step: int
    params: np.ndarray                    # flattened parameter vector θ
    gradient: np.ndarray                  # flattened gradient ∇ℒ(θ)
    learning_rate: float
    loss: float | None = None
    constraints: Sequence[Constraint] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class OthismosCallback(Protocol):
    """The minimal protocol every callback must satisfy.

    Framework-specific adapters (PyTorch, HF, etc.) implement this
    via duck typing. No inheritance required — just match the methods.
    """

    def on_step_end(self, ctx: StepContext, measurement: PressureMeasurement) -> None: ...
    def on_epoch_end(self, ctx: StepContext, gauge: PressureGauge) -> None: ...
    def on_train_end(self, ctx: StepContext, gauge: PressureGauge) -> None: ...


class CallbackBundle:
    """Compose multiple callbacks into one.

    Usage:
        bundle = CallbackBundle([logging_cb, wandb_cb, early_stop_cb])
        # bundle.on_step_end(...) calls each child in order
    """

    def __init__(self, callbacks: list[OthismosCallback]):
        self._callbacks = callbacks

    def on_step_end(self, ctx: StepContext, measurement: PressureMeasurement) -> None:
        for cb in self._callbacks:
            cb.on_step_end(ctx, measurement)

    def on_epoch_end(self, ctx: StepContext, gauge: PressureGauge) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(ctx, gauge)

    def on_train_end(self, ctx: StepContext, gauge: PressureGauge) -> None:
        for cb in self._callbacks:
            cb.on_train_end(ctx, gauge)
```

### 1.2 PyTorch Integration (Native — Optimizer Hooks)

PyTorch provides `optimizer.register_step_pre_hook` and `optimizer.register_step_post_hook`. The óthismos integration uses the **post-hook** to measure pressure after the optimizer computes its step but before/after application.

```python
# src/othismos/integrations/torch.py (PROPOSED)

from __future__ import annotations
from typing import Sequence
import numpy as np
import torch

from othismos.pressure import (
    Constraint, PressureGauge, PressureMeasurement,
    compute_othismos,
)
from othismos.callbacks import StepContext, OthismosCallback, CallbackBundle
from othismos.phases import MoltCycleTracker
from othismos.diagnostics import PopcornDiagnostic


class TorchConstraintAdapter:
    """Adapt PyTorch parameter constraints to óthismos Constraint objects.

    PyTorch doesn't have native Constraint objects, so we provide
    adapters for common patterns: weight decay clips, gradient clipping,
    custom projection layers.
    """

    @staticmethod
    def from_optimizer(
        optimizer: torch.optim.Optimizer,
        max_norm: float | None = None,
    ) -> list[Constraint]:
        """Extract constraints from an optimizer's configuration.

        Handles:
        - weight_decay → L2 ball constraint
        - max_norm (from clip_grad_norm_) → gradient norm constraint
        """
        from othismos.pressure import l2_constraint

        constraints: list[Constraint] = []

        for group in optimizer.param_groups:
            wd = group.get("weight_decay", 0.0)
            if wd > 0:
                # Approximate: weight decay creates an implicit L2 ball
                # whose radius depends on the regularization strength
                constraints.append(
                    l2_constraint(
                        name=f"weight_decay_{wd}",
                        radius=1.0 / wd,  # heuristic
                    )
                )

        return constraints

    @staticmethod
    def from_param_clipping(
        params: list[torch.nn.Parameter],
        min_val: float,
        max_val: float,
    ) -> list[Constraint]:
        """Create box constraints for clipped parameters."""
        from othismos.pressure import box_constraint

        constraints = []
        for i, p in enumerate(params):
            n = p.numel()
            constraints.append(
                box_constraint(
                    name=f"clip_param_{i}",
                    lows=np.full(n, min_val),
                    highs=np.full(n, max_val),
                )
            )
        return constraints


def _flatten(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().ravel()


def attach_to_optimizer(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    constraints: Sequence[Constraint],
    callbacks: OthismosCallback | CallbackBundle | None = None,
    track_phases: bool = True,
    run_diagnostics: bool = True,
) -> PressureGauge:
    """Attach óthismos instrumentation to a PyTorch optimizer.

    This registers pre/post hooks on the optimizer that:
    1. Capture the desired step (pre-hook)
    2. Measure actual step after projection (post-hook)
    3. Dispatch to callbacks

    Returns the PressureGauge for direct inspection.

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(10, 1)
        >>> opt = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> gauge = attach_to_optimizer(opt, model, constraints=[...])
        >>> # Now train normally — óthismos measures automatically

    """
    gauge = PressureGauge()
    bundle = callbacks if isinstance(callbacks, CallbackBundle) else (
        CallbackBundle([callbacks]) if callbacks is not None else CallbackBundle([])
    )

    phase_tracker = MoltCycleTracker() if track_phases else None
    diagnostic = PopcornDiagnostic() if run_diagnostics else None

    # Capture pre-step state
    _pre_step_state: dict = {}

    def pre_hook(opt, args, kwargs):
        # Snapshot params and grads BEFORE the step
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        grads = torch.cat([
            (p.grad.view(-1) if p.grad is not None
             else torch.zeros_like(p.data.view(-1)))
            for p in model.parameters()
        ])
        lr = opt.param_groups[0]["lr"]
        _pre_step_state["theta"] = _flatten(params)
        _pre_step_state["gradient"] = _flatten(grads)
        _pre_step_state["lr"] = lr

    def post_hook(opt, args, kwargs):
        if "theta" not in _pre_step_state:
            return

        theta_old = _pre_step_state["theta"]
        gradient = _pre_step_state["gradient"]
        lr = _pre_step_state["lr"]

        params_new = torch.cat([p.data.view(-1) for p in model.parameters()])
        theta_new = _flatten(params_new)

        # Direct measurement: compare desired vs actual step
        desired = -lr * gradient
        actual = theta_new - theta_old
        violation = desired - actual
        pressure = float(np.linalg.norm(violation))

        measurement = PressureMeasurement(
            step=gauge._step,
            desired_step=desired,
            actual_step=actual,
            violation=violation,
            pressure=pressure,
        )
        gauge._history.append(measurement)
        gauge._step += 1

        # Phase tracking
        reading = None
        if phase_tracker:
            reading = phase_tracker.update(pressure)

        # Diagnostics
        diag = None
        if diagnostic:
            heat = lr * float(np.linalg.norm(gradient))
            pressures = [m.pressure for m in gauge._history]
            diag = diagnostic.diagnose(pressures, heat=heat)

        # Dispatch to callbacks
        ctx = StepContext(
            step=measurement.step,
            params=theta_new,
            gradient=gradient,
            learning_rate=lr,
            constraints=list(constraints),
            extra={
                "phase_reading": reading,
                "diagnostic": diag,
            },
        )
        bundle.on_step_end(ctx, measurement)

    optimizer.register_step_pre_hook(pre_hook)
    optimizer.register_step_post_hook(post_hook)

    return gauge
```

### 1.3 HuggingFace Trainer Integration

HuggingFace's `TrainerCallback` is the natural integration point. Óthismos provides a `TrainerCallback` subclass that hooks into the HF training lifecycle.

```python
# src/othismos/integrations/huggingface.py (PROPOSED)

from __future__ import annotations
import numpy as np

from othismos.pressure import (
    Constraint, PressureGauge, compute_othismos,
)
from othismos.phases import MoltCycleTracker
from othismos.diagnostics import PopcornDiagnostic


class OthismosTrainerCallback:
    """HuggingFace TrainerCallback for óthismos measurement.

    Usage with HF Trainer:
        >>> from transformers import Trainer, TrainingArguments
        >>> othismos_cb = OthismosTrainerCallback(constraints=[...])
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_ds,
        ...     callbacks=[othismos_cb],
        ... )
        >>> trainer.train()
        >>> # Access measurements:
        >>> othismos_cb.gauge.history
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        track_phases: bool = True,
        run_diagnostics: bool = True,
        log_to_wandb: bool = False,
        log_to_tensorboard: bool = False,
    ):
        self.constraints = constraints or []
        self.gauge = PressureGauge(window_size=10000)
        self.phase_tracker = MoltCycleTracker() if track_phases else None
        self.diagnostic = PopcornDiagnostic() if run_diagnostics else None
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard

    # HF callback interface — duck-typed, no inheritance needed
    def on_init_end(self, args, state, control, **kwargs):
        """Called when Trainer finishes initialization."""
        self._model = kwargs.get("model")

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each optimizer.step()."""
        model = kwargs.get("model", getattr(self, "_model", None))
        if model is None:
            return

        # Flatten params and grads
        import torch
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        grads = torch.cat([
            (p.grad.view(-1) if p.grad is not None
             else torch.zeros_like(p.data.view(-1)))
            for p in model.parameters()
        ])

        theta = params.detach().cpu().numpy()
        gradient = grads.detach().cpu().numpy()
        lr = state.learning_rate if hasattr(state, "learning_rate") else args.learning_rate

        m = self.gauge.measure(theta, gradient, lr, self.constraints)

        # Phase + diagnostics
        phase_reading = None
        if self.phase_tracker:
            phase_reading = self.phase_tracker.update(m.pressure)

        diag = None
        if self.diagnostic:
            heat = lr * float(np.linalg.norm(gradient))
            pressures = [mm.pressure for mm in self.gauge._history]
            diag = self.diagnostic.diagnose(pressures, heat=heat)

        # Logging
        logs = {
            "othismos/pressure": m.pressure,
            "othismos/mean_pressure": self.gauge.mean_pressure,
            "othismos/trend": self.gauge.pressure_trend,
        }
        if phase_reading:
            logs["othismos/phase"] = phase_reading.phase.value
            logs["othismos/phase_confidence"] = phase_reading.confidence
        if diag:
            logs["othismos/health"] = diag.health.value
            logs["othismos/efficiency"] = diag.pressure_efficiency

        # HF Trainer handles wandb/tensorboard routing via logs dict
        # But we can also log directly:
        if self.log_to_wandb:
            try:
                import wandb
                wandb.log(logs, step=state.global_step)
            except ImportError:
                pass

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        zone = self.gauge.goldilocks()
        logs = {
            "othismos/goldilocks_low": zone.lower_bound,
            "othismos/goldilocks_high": zone.upper_bound,
            "othismos/cycle_count": (
                self.phase_tracker.cycle_count if self.phase_tracker else 0
            ),
        }
        if self.phase_tracker and self.phase_tracker.cycles:
            staircase = self.phase_tracker.staircase_metric()
            logs["othismos/periodicity"] = staircase.get("periodicity", 0)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Final report at training completion."""
        if self.phase_tracker:
            staircase = self.phase_tracker.staircase_metric()
            print(f"\n[Óthismos] Training complete.")
            print(f"  Total cycles: {staircase.get('cycles', 0)}")
            print(f"  Health: {staircase.get('health', 'unknown')}")
            print(f"  Mean pressure: {self.gauge.mean_pressure:.6f}")
            print(f"  Goldilocks zone: [{self.gauge.goldilocks().lower_bound:.4f}, "
                  f"{self.gauge.goldilocks().upper_bound:.4f}]")
        return control
```

### 1.4 PyTorch Lightning Integration

Lightning's `Callback` class provides hooks like `on_train_batch_end`, `on_train_epoch_end`. The óthismos adapter is straightforward:

```python
# src/othismos/integrations/lightning.py (PROPOSED)

from __future__ import annotations
import numpy as np
import torch

from othismos.pressure import Constraint, PressureGauge
from othismos.phases import MoltCycleTracker
from othismos.diagnostics import PopcornDiagnostic


class OthismosLightningCallback:
    """PyTorch Lightning callback for óthismos pressure measurement.

    Usage:
        >>> from lightning.pytorch import Trainer
        >>> from othismos.integrations.lightning import OthismosLightningCallback
        >>> trainer = Trainer(callbacks=[OthismosLightningCallback()])
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        track_phases: bool = True,
    ):
        self.constraints = constraints or []
        self.gauge = PressureGauge(window_size=10000)
        self.phase_tracker = MoltCycleTracker() if track_phases else None
        self._theta_prev: np.ndarray | None = None

    def _flatten_params(self, model) -> np.ndarray:
        return torch.cat([p.data.view(-1) for p in model.parameters()]) \
                     .cpu().numpy()

    def _flatten_grads(self, model) -> np.ndarray:
        return torch.cat([
            (p.grad.view(-1) if p.grad is not None
             else torch.zeros_like(p.data.view(-1)))
            for p in model.parameters()
        ]).cpu().numpy()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Measure pressure after each training batch."""
        model = pl_module
        theta = self._flatten_params(model)
        gradient = self._flatten_grads(model)
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        if self._theta_prev is not None:
            # Measure: desired step vs what actually happened
            desired = -lr * gradient
            actual = theta - self._theta_prev
            violation = desired - actual
            pressure = float(np.linalg.norm(violation))

            from othismos.pressure import PressureMeasurement
            m = PressureMeasurement(
                step=trainer.global_step,
                desired_step=desired,
                actual_step=actual,
                violation=violation,
                pressure=pressure,
            )
            self.gauge._history.append(m)
            self.gauge._step += 1

            if self.phase_tracker:
                self.phase_tracker.update(pressure)

            # Log to Lightning's logging system
            pl_module.log("othismos/pressure", pressure, prog_bar=True)
            pl_module.log("othismos/mean_pressure", self.gauge.mean_pressure)

        self._theta_prev = theta

    def on_train_epoch_end(self, trainer, pl_module):
        zone = self.gauge.goldilocks()
        pl_module.log("othismos/goldilocks_low", zone.lower_bound)
        pl_module.log("othismos/goldilocks_high", zone.upper_bound)
        if self.phase_tracker:
            pl_module.log("othismos/cycles", self.phase_tracker.cycle_count)
```

### 1.5 Weights & Biases / TensorBoard Logging

Rather than hardcoding logging, óthismos provides a generic logging callback that adapts to any backend:

```python
# src/othismos/integrations/logging.py (PROPOSED)

from __future__ import annotations
from typing import Any
from othismos.callbacks import StepContext
from othismos.pressure import PressureGauge, PressureMeasurement


class LoggingCallback:
    """Generic metric logger — adapts to wandb, tensorboard, mlflow, etc.

    The log_fn abstraction means óthismos never imports a logging
    library directly. The user provides the function.
    """

    def __init__(
        self,
        log_fn: callable,  # (metrics: dict, step: int) -> None
        prefix: str = "othismos",
        log_every: int = 1,  # log every N steps
    ):
        self.log_fn = log_fn
        self.prefix = prefix
        self.log_every = log_every

    def on_step_end(self, ctx: StepContext, measurement: PressureMeasurement) -> None:
        if ctx.step % self.log_every != 0:
            return
        self.log_fn({
            f"{self.prefix}/pressure": measurement.pressure,
            f"{self.prefix}/is_pushing": float(measurement.is_pushing),
        }, ctx.step)

    def on_epoch_end(self, ctx: StepContext, gauge: PressureGauge) -> None:
        self.log_fn({
            f"{self.prefix}/mean_pressure": gauge.mean_pressure,
            f"{self.prefix}/trend": gauge.pressure_trend,
            f"{self.prefix}/goldilocks_low": gauge.goldilocks().lower_bound,
            f"{self.prefix}/goldilocks_high": gauge.goldilocks().upper_bound,
        }, ctx.step)

    def on_train_end(self, ctx: StepContext, gauge: PressureGauge) -> None:
        self.log_fn({
            f"{self.prefix}/final_pressure": gauge.current_pressure,
            f"{self.prefix}/final_mean": gauge.mean_pressure,
        }, ctx.step)


# --- Ready-to-use adapters ---

def wandb_logger(project: str | None = None, **wandb_kwargs):
    """Create a W&B log_fn for LoggingCallback."""
    import wandb
    def log_fn(metrics: dict, step: int) -> None:
        wandb.log(metrics, step=step)
    return log_fn


def tensorboard_logger(log_dir: str):
    """Create a TensorBoard log_fn for LoggingCallback."""
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    def log_fn(metrics: dict, step: int) -> None:
        for key, val in metrics.items():
            writer.add_scalar(key, val, step)
    return log_fn


# Usage:
#   import wandb; wandb.init(project="my-exp")
#   cb = LoggingCallback(wandb_logger(), prefix="ot")
#   gauge = attach_to_optimizer(opt, model, constraints, callbacks=cb)
```

### 1.6 Ray Tune Integration

Ray Tune uses `session.report()` for function-based trainables and `Trainable` class for class-based. Óthismos integrates by measuring inside the training function and reporting pressure as a metric:

```python
# Example: using óthismos with Ray Tune

import ray.tune as tune
from othismos.pressure import PressureGauge, l2_constraint
from othismos.phases import MoltCycleTracker

def train_fn(config):
    # ... set up model, optimizer ...

    gauge = PressureGauge()
    tracker = MoltCycleTracker()
    constraint = l2_constraint("budget", radius=config["radius"])

    for epoch in range(config["epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            # Measure óthismos
            theta = flatten_params(model)
            gradient = flatten_grads(model)
            m = gauge.measure(theta, gradient, lr, [constraint])
            tracker.update(m.pressure)

        # Report to Ray Tune
        tune.report(
            loss=epoch_loss,
            othismos_pressure=gauge.mean_pressure,
            othismos_phase=tracker.current_phase.value if tracker.current_phase else 0,
            othismos_goldilocks_low=gauge.goldilocks().lower_bound,
            othismos_goldilocks_high=gauge.goldilocks().upper_bound,
        )

# Ray Tune can then optimize for pressure-aware objectives:
tuner = tune.Tuner(
    train_fn,
    param_space={
        "radius": tune.uniform(0.1, 2.0),
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": 10,
    },
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        # Could also optimize for "othismos_pressure" to find
        # configs that maintain productive pressure
    ),
)
tuner.fit()
```

---

## 2. API Audit

### 2.1 What's Good

| Aspect | Assessment |
|--------|-----------|
| **Domain modeling** | Excellent. PressureMeasurement, Constraint, MoltPhase, PopcornDiagnostic — the abstractions are clean and map to the math/essays. |
| **Dataclasses** | Good use of `@dataclass` for immutable value objects (PressureMeasurement, GoldilocksZone, PhaseReading, DiagnosticResult, Deposit, MoltCycle). |
| **Enums** | IntEnum for phases, Enum for health/constraint types. Correct choices. |
| **Numpy-only dep** | Minimal dependency surface. Good. |
| **Separation of concerns** | pressure / phases / diagnostics / ecology are well-separated modules. |
| **Property-driven API** | `is_pushing`, `is_healthy`, `is_orphan`, `description`, `width` — pythonic, readable. |
| **Factory functions** | `l2_constraint()`, `box_constraint()` are the right pattern (easier than subclassing). |
| **Tests** | Good coverage of core pressure computation. Edge cases tested. |

### 2.2 What's Awkward

#### A. `callable` type annotations

**Problem:** `Constraint.bound_fn`, `project_fn`, `normal_fn` use bare `callable` (lowercase) which is not a proper type annotation in modern Python.

```python
# CURRENT (pressure.py)
@dataclass
class Constraint:
    name: str
    type: ConstraintType
    bound_fn: callable          # ← bare callable, not typed
    project_fn: callable        # ← no signature information
    normal_fn: callable | None = None
```

**Fix:** Use `typing.Callable` with proper signatures, or define a `Protocol`:

```python
from typing import Protocol

class ProjectionFn(Protocol):
    def __call__(self, theta: np.ndarray) -> np.ndarray: ...

class FeasibilityFn(Protocol):
    def __call__(self, theta: np.ndarray) -> bool: ...

@dataclass
class Constraint:
    name: str
    type: ConstraintType
    bound_fn: FeasibilityFn
    project_fn: ProjectionFn
    normal_fn: ProjectionFn | None = None
```

#### B. `compute_othismos` returns step=-1

**Problem:** The function hardcodes `step=-1` and expects the caller to override. This is surprising and fragile.

```python
# CURRENT
return PressureMeasurement(step=-1, ...)  # caller can override
```

**Fix:** Add `step` as a parameter with a sensible default:

```python
def compute_othismos(
    theta: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    constraints: Sequence[Constraint],
    step: int = 0,  # ← explicit, not -1
) -> PressureMeasurement:
```

#### C. `PressureGauge._history` and `._step` are private but accessed by integrations

**Problem:** The HuggingFace and Lightning integrations need `gauge._history` and `gauge._step` — accessing private members. The `history` property returns a copy, which is safe but doesn't allow the integrations to append efficiently.

**Fix:** Expose protected methods:

```python
class PressureGauge:
    def record(self, measurement: PressureMeasurement) -> None:
        """Record an externally-computed measurement."""
        self._history.append(measurement)
        self._step += 1
        if len(self._history) > self._window:
            self._history = self._history[-self._window:]

    @property
    def raw_history(self) -> list[PressureMeasurement]:
        """Direct reference to internal history (read-write for integrations)."""
        return self._history
```

#### D. No serialization support

**Problem:** None of the dataclasses support serialization (JSON, pickle-safe, dict conversion). `PressureMeasurement` contains numpy arrays which aren't JSON-serializable by default.

**Fix:** Add `to_dict()` / `from_dict()` methods, and make the dataclasses picklable:

```python
@dataclass
class PressureMeasurement:
    ...

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "pressure": self.pressure,
            "is_pushing": self.is_pushing,
            "pressure_by_constraint": dict(self.pressure_by_constraint),
            "desired_step": self.desired_step.tolist(),
            "actual_step": self.actual_step.tolist(),
            "violation": self.violation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PressureMeasurement":
        return cls(
            step=d["step"],
            desired_step=np.array(d["desired_step"]),
            actual_step=np.array(d["actual_step"]),
            violation=np.array(d["violation"]),
            pressure=d["pressure"],
            pressure_by_constraint=d.get("pressure_by_constraint", {}),
        )
```

#### E. No async support

**Problem:** Some users (distributed training, async optimizers) may want async measurement. Currently all APIs are synchronous.

**Assessment:** Not urgent. The core computation is pure math — it doesn't need I/O. Async is better handled at the callback/integration layer. Document this as a non-goal for v1.

#### F. `GoldilocksZone` is not exported from `__init__.py`

**Problem:** `goldilocks_range()` returns a `GoldilocksZone` but users can't import it directly.

```python
# CURRENT __init__.py
from othismos.pressure import (
    PressureMeasurement,
    PressureGauge,
    compute_othismos,
    goldilocks_range,
)
# GoldilocksZone is MISSING
```

**Fix:** Add to exports:

```python
from othismos.pressure import GoldilocksZone  # add this
```

Same for `Constraint`, `ConstraintType`, `l2_constraint`, `box_constraint`, `PhaseReading`.

#### G. `Reef.search()` is O(n) string matching

**Problem:** Linear scan with `.lower()` on every deposit. Won't scale.

**Assessment:** Acceptable for v1 (reef is meant for small graphs). Document performance characteristics. For v2, consider an inverted index or integration with a proper search backend.

#### H. `Reef._compute_depth` can stack-overflow on deep citation chains

**Problem:** Recursive depth computation with manual memoization. Python's default recursion limit (1000) could be hit.

**Fix:** Convert to iterative DFS, or at minimum add `sys.setrecursionlimit` safety:

```python
def _compute_depth(self, deposit_id: str) -> float:
    """Iterative computation to avoid stack overflow."""
    memo: dict[str, float] = {}
    stack = [(deposit_id, False)]

    while stack:
        node_id, processed = stack.pop()
        if node_id in memo:
            continue
        deposit = self._deposits.get(node_id)
        if deposit is None:
            memo[node_id] = 0.0
            continue

        if not deposit.referenced_by:
            memo[node_id] = 1.0
            continue

        if not processed:
            stack.append((node_id, True))
            for child_id in deposit.referenced_by:
                if child_id not in memo:
                    stack.append((child_id, False))
        else:
            child_depths = sum(memo.get(c, 0.0) for c in deposit.referenced_by)
            depth = 1.0 + child_depths / max(len(deposit.referenced_by), 1)
            memo[node_id] = depth

    return memo.get(deposit_id, 0.0)
```

### 2.3 What's Missing

| Feature | Priority | Notes |
|---------|----------|-------|
| **Protocol classes** | High | Define `typing.Protocol` for callables, callbacks |
| **Serialization** | High | `to_dict()` / `from_dict()` on all dataclasses |
| **Constraint protocol** | High | Users need to know what signature their custom constraint must match |
| **Flatten/unflatten utilities** | High | Converting between model params and flat vectors is ubiquitous |
| **Async-safe gauge** | Medium | For distributed training (DDP, FSDP) |
| **Config object** | Medium | Single dataclass to configure gauge + tracker + diagnostic together |
| **Pandas export** | Medium | `gauge.to_dataframe()` for analysis |
| **Visualization helpers** | Medium | `plot_pressure(gauge)`, `plot_molt_cycle(tracker)` |
| **Gradient estimation** | Low | Finite-difference gradient when analytical isn't available |

### 2.4 Type Annotation Improvements

```python
# CURRENT: bare callable
bound_fn: callable

# RECOMMENDED: proper typing
from typing import Protocol, runtime_checkable

@runtime_checkable
class FeasibilityFn(Protocol):
    def __call__(self, theta: np.ndarray) -> bool: ...

@runtime_checkable
class ProjectionFn(Protocol):
    def __call__(self, theta: np.ndarray) -> np.ndarray: ...

# Use in Constraint:
@dataclass(slots=True)  # also: add slots for memory efficiency
class Constraint:
    name: str
    type: ConstraintType
    bound_fn: FeasibilityFn
    project_fn: ProjectionFn
    normal_fn: ProjectionFn | None = None
```

Also recommend adding `@dataclass(slots=True)` (Python 3.10+) to all dataclasses — reduces memory by ~40% and improves attribute access speed.

---

## 3. Packaging & Distribution

### 3.1 Current State

The `pyproject.toml` is already good:
- ✅ Uses PEP 621 metadata
- ✅ `setuptools` build backend
- ✅ `requires-python >= 3.10`
- ✅ MIT license
- ✅ Optional `[dev]` extras
- ✅ Proper classifiers

### 3.2 Recommended Changes

#### Split optional dependencies

```toml
[project.optional-dependencies]
# Core extras
torch = ["torch>=2.0"]           # PyTorch integration
viz = ["matplotlib>=3.7"]        # Visualization helpers
pandas = ["pandas>=2.0"]         # DataFrame export
logging = ["wandb>=0.15"]        # W&B logging (optional)

# Convenience
all = [
    "othismos[torch,viz,pandas]",
]

# Development
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-benchmark>=4.0",      # performance tests
    "mypy>=1.0",                  # type checking
    "ruff>=0.3",                  # linting + formatting
]

docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
```

#### Add project metadata

```toml
[project.urls]
Homepage = "https://github.com/SuperInstance/othismos"
Repository = "https://github.com/SuperInstance/othismos"
Documentation = "https://superinstance.github.io/othismos"
"Bug Tracker" = "https://github.com/SuperInstance/othismos/issues"
Changelog = "https://github.com/SuperInstance/othismos/blob/main/CHANGELOG.md"
```

#### Entry points (for CLI)

```toml
[project.scripts]
othismos = "othismos.cli:main"   # CLI for pressure analysis
```

### 3.3 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml (PROPOSED)
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      - run: mypy src/othismos/
      - run: pytest --cov=othismos --cov-report=xml
      - uses: codecov/codecov-action@v4

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # trusted publishing
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### 3.4 PyPI Publishing Checklist

- [ ] Choose license (MIT is already set)
- [ ] Add `LICENSE` file in repo root
- [ ] Set up [trusted publishing](https://docs.pypi.org/trusted-publishers/) (no API tokens needed)
- [ ] Tag releases: `git tag v0.1.0 && git push --tags`
- [ ] Add `CHANGELOG.md`
- [ ] Verify `python -m build` produces clean wheel + sdist
- [ ] Test on TestPyPI first: `twine upload --repository testpypi dist/*`
- [ ] Long description renders correctly (README.md as `readme`)

### 3.5 Documentation Site

Use MkDocs Material (matches the existing aesthetic):

```yaml
# mkdocs.yml (PROPOSED)
site_name: Óthismos
site_description: "The force a bounded system exerts against its bounds."
theme:
  name: material
  features:
    - navigation.tabs
    - code.copy
    - content.code.annotate

nav:
  - Home: index.md
  - Guide:
    - Getting Started: guide/getting-started.md
    - Pressure Measurement: guide/pressure.md
    - Molt Cycle: guide/phases.md
    - Popcorn Diagnostic: guide/diagnostics.md
    - Reef Ecology: guide/ecology.md
  - Integrations:
    - PyTorch: integrations/pytorch.md
    - HuggingFace: integrations/huggingface.md
    - Lightning: integrations/lightning.md
    - W&B / TensorBoard: integrations/logging.md
    - Ray Tune: integrations/ray-tune.md
  - API Reference: reference/
  - Theory:
    - Math: theory/math.md
    - Essays: theory/essays.md

plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: false
            show_root_heading: true
```

### 3.6 Version Strategy

```python
# Use semver: MAJOR.MINOR.PATCH
# 0.x.x = alpha (current: 0.1.0)
# 1.0.0 = first stable release
#     - Public API stabilized
#     - All framework integrations working
#     - Documentation complete
#     - At least 90% test coverage
```

Consider `setuptools-scm` for automatic versioning from git tags:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools-scm>=8.0", "setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
dynamic = ["version"]

[tool.setuptools_scm]
fallback_version = "0.1.0"
```

---

## 4. Comparison with Similar Libraries

### 4.1 `torch.optim` (constraint clipping, weight decay)

**What it does:** PyTorch's optimizer ecosystem handles constraints implicitly — weight decay, gradient clipping, param clamp