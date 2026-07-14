# Óthismos Code Review — Hostile Audit

**Reviewer:** GLM-5.2 (MiniMax subagent)
**Date:** 2026-07-14
**Repo version:** v0.3.0
**Scope:** All 13 source modules (~1,800 LoC), plus `protocols.py` and `__init__.py`

---

## Methodology

I read every line of every source file. I looked for bugs that will destroy user data, APIs that lie about what they do, numeric instabilities, performance landmines, and design decisions that will cause silent wrong answers in production. I did not go easy on this code.

---

## Summary by Severity

| Severity | Count | Summary |
|---|---|---|
| CRITICAL | 7 | Silent wrong answers, dimension mismatches, broken serialization, unbounded memory |
| HIGH | 10 | Misleading metrics, order-dependent math, fragile coupling, missing validation |
| MEDIUM | 8 | API inconsistencies, performance issues, questionable design |
| LOW | 5 | Minor robustness, documentation, style |

---

## CRITICAL

### C1. `l2_constraint` default `center=np.zeros(1)` — shape mismatch with multi-dimensional theta

**File:** `src/othismos/pressure.py:92`
**Severity:** CRITICAL

```python
center = center if center is not None else np.zeros(1)
```

When the user calls `l2_constraint("w", radius=1.0)` without a center, the default is a 1-element array. For 1-D `theta` of any length > 1, broadcasting saves you. But for 2-D `theta` (e.g., a weight matrix flattened differently), `np.linalg.norm(theta - center)` computes the Frobenius norm over the entire array — which happens to be correct for a global L2 ball, BUT the `bound_fn` and `project_fn` closures capture this `np.zeros(1)` object. If a user later modifies the center array in-place (unlikely but possible), the constraint silently changes.

More critically: the `normal_fn` computes `(theta - center) / max(np.linalg.norm(theta - center), 1e-12)`. When `theta == center` (i.e., at the center of the ball), this returns a zero vector as the "normal," which is mathematically undefined and will produce NaN if used in any dot product.

**Fix:**
```python
def l2_constraint(name: str, radius: float, center: np.ndarray | None = None) -> Constraint:
    if center is None:
        # Use 0.0 scalar — broadcasts correctly to any shape
        center_value = 0.0
        bound_fn = lambda theta: np.linalg.norm(theta - center_value) <= radius
        project_fn = lambda theta: _l2_ball_project(theta, np.broadcast_to(center_value, theta.shape), radius)
        normal_fn = lambda theta: (
            (d := theta - center_value) / max(np.linalg.norm(d), 1e-12)
        )
    else:
        center = np.asarray(center)
        bound_fn = lambda theta: np.linalg.norm(theta - center) <= radius
        project_fn = lambda theta: _l2_ball_project(theta, center, radius)
        normal_fn = lambda theta: (
            (d := theta - center) / max(np.linalg.norm(d), 1e-12)
        )
    ...
```

---

### C2. Sequential constraint application is order-dependent — and the per-constraint pressure breakdown lies

**File:** `src/othismos/pressure.py:115-130`

```python
for constraint in constraints:
    projected = constraint.project(theta_new)
    violation_component = theta_new - projected
    individual_pressure = float(np.linalg.norm(violation_component))
    if individual_pressure > 1e-12:
        pressure_by[constraint.name] = individual_pressure
    theta_new = projected
```

Constraints are applied sequentially. This means:

1. **The total pressure depends on constraint ordering.** Applying `[A, B]` gives a different result than `[B, A]` because the first projection changes the input to the second.

2. **The per-constraint breakdown is misleading.** `pressure_by[A]` includes the shadow of B's future projection. If A clips 0.1 units and B clips 0.5 units, but applying A first moves theta into a region where B clips even more, the numbers don't add up to the total.

3. **The `pressure_by_constraint` values will be silently wrong** for any analysis that treats them as independent contributions.

This is a fundamental mathematical issue: projecting onto the intersection of convex sets via alternating projections (POCS) is only guaranteed to converge in the limit, not in one pass.

**Fix (document the limitation):**
```python
def compute_othismos(
    theta, gradient, learning_rate, constraints
) -> PressureMeasurement:
    """
    ...
    
    .. note::
        Constraints are applied sequentially (alternating projection).
        The per-constraint breakdown reflects marginal contribution given
        prior projections, NOT independent contribution. For accurate
        decomposition, use mutually orthogonal constraint sets or Dykstra's
        algorithm.
    """
```

**Fix (if correct decomposition matters):** Apply each constraint independently to `theta + desired`, measure each violation separately, then compute the combined projection.

---

### C3. `OthismosTorchCallback.post_step` computes pressure wrong — double-adds pre-step params

**File:** `src/othismos/integrations.py:138-150`

```python
# The actual step taken
actual_step = post_params - self._pre_step_params

# Compute what the unconstrained step would have been
projected = self._pre_step_params.copy()
for c in self.constraints:
    projected = c.project(self._pre_step_params + actual_step)
```

`self._pre_step_params + actual_step` = `pre + (post - pre)` = `post`. So the constraints are projected onto `post_params`, which is correct for measuring post-hoc pressure. BUT the `PressureMeasurement` is then constructed with:

```python
m = PressureMeasurement(
    step=self._step,
    desired_step=actual_step,          # ← This is the optimizer's step, NOT the unconstrained desired step
    actual_step=projected - self._pre_step_params,  # ← This is projected_pos - pre = projected step
    violation=violation,
    pressure=pressure,
)
```

The `desired_step` field is labeled "the unconstrained gradient step" in the dataclass, but here it's set to the **optimizer's actual step** (which already includes optimizer-specific dynamics like momentum). The naming is wrong and the measurement is misleading.

Furthermore, the callback never writes the projected parameters back to the model. It measures pressure but doesn't enforce constraints. This is only mentioned in the docstring of the class ("via projection") but the actual projection is never applied. The model trains unconstrained while the callback pretends to measure constraint pressure.

**Fix:** Either (a) rename `desired_step` to `optimizer_step` and document that this is observatory-only, or (b) write back the projected params and measure the real violation:

```python
# Option (b): actually enforce constraints
projected_params = post_params.copy()
for c in self.constraints:
    projected_params = c.project(projected_params)

violation = post_params - projected_params
pressure = float(np.linalg.norm(violation))

# Write back to model
self._write_back_params(model, projected_params)
```

---

### C4. `OthismosTorchCallback` flattens entire model to numpy — OOM on any real model

**File:** `src/othismos/integrations.py:101-107`

```python
def _flatten_params(self, model) -> np.ndarray:
    import torch
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
```

For a 7B parameter model (float32), this creates a ~28GB numpy array. Two of them (pre and post step) = ~56GB. The `_write_back_params` creates another copy. On any model larger than ~1B parameters, this will OOM.

Even worse, `p.detach().cpu().numpy()` creates a copy of each parameter and moves it to CPU. For GPU models, this is a synchronous D2H copy of the entire model, twice per step, blocking the training loop.

**Fix:** Document the limitation and add chunked/streaming pressure computation, or compute pressure in parameter-space using torch tensors directly:

```python
def _compute_pressure_torch(self, model) -> tuple[float, dict]:
    """Compute pressure using torch tensors — no CPU transfer needed."""
    import torch
    total_violation = torch.tensor(0.0, device=next(model.parameters()).device)
    for c in self.constraints:
        # Apply constraint to flattened params in chunks
        ...
```

---

### C5. `save_history` serializes full numpy arrays as JSON — will crash or produce gigabytes

**File:** `src/othismos/serialization.py:37-54`

For a model with N parameters, each `PressureMeasurement` stores `desired_step`, `actual_step`, and `violation` as full numpy arrays of length N. `save_history` calls `.tolist()` on each, producing JSON arrays of N floats. For 1000 measurements of a 1M-parameter model, that's 3 billion floats in a JSON file.

Even for modest models (100K params, 100 steps), the JSON file will be hundreds of MB and slow to parse.

**Fix:** Use a binary format (NPZ/HDF5) for array data, or only serialize the pressure scalar and constraint breakdown:

```python
def save_history(gauge: PressureGauge, path: str | Path, include_arrays: bool = False) -> None:
    data = {
        "version": "0.1.0",
        "step_count": gauge._step,
        "measurements": [
            {
                "step": m.step,
                "pressure": float(m.pressure),
                "pressure_by_constraint": m.pressure_by_constraint,
                **({"desired_step": _ndarray_to_list(m.desired_step),
                    "actual_step": _ndarray_to_list(m.actual_step),
                    "violation": _ndarray_to_list(m.violation)}
                   if include_arrays else {})
            }
            for m in gauge.history
        ],
    }
    ...
```

---

### C6. `PhaseClassifier` uses `or` for threshold auto-calibration — `crisis_threshold=0.0` silently ignored

**File:** `src/othismos/phases.py:88-89`

```python
crisis_th = self.crisis_threshold or (mean_p + 2 * std_p if std_p > 0 else mean_p * 1.5)
expansion_th = self.expansion_floor or (mean_p * 0.3)
```

If a user explicitly sets `crisis_threshold=0.0` (meaning "any positive pressure is crisis"), `0.0 or X` evaluates to `X`. The threshold is silently overridden. Same for `expansion_floor=0.0`.

This is the classic Python `or` vs `is None` bug. It's particularly insidious because 0.0 is a meaningful threshold value (it means "crisis at any pressure").

**Fix:**
```python
crisis_th = self.crisis_threshold if self.crisis_threshold is not None else (
    mean_p + 2 * std_p if std_p > 0 else mean_p * 1.5
)
expansion_th = self.expansion_floor if self.expansion_floor is not None else (mean_p * 0.3)
```

---

### C7. `Reef._compute_depth` produces wrong results on cyclic reference graphs

**File:** `src/othismos/ecology.py:195-210`

```python
def _compute_depth(self, deposit_id: str, memo: dict | None = None) -> float:
    ...
    # Prevent infinite recursion on cycles
    memo[deposit_id] = 1.0  # placeholder
    child_depths = sum(self._compute_depth(child, memo) for child in deposit.referenced_by)
    depth = 1.0 + child_depths / max(len(deposit.referenced_by), 1)
    memo[deposit_id] = depth
    return depth
```

The placeholder `memo[deposit_id] = 1.0` prevents infinite recursion, but it means that any node in a cycle gets depth 1.0 as its contribution to its parents. This produces incorrect depth scores for any graph with cycles (which the `referenced_by` set can create if deposits reference each other — there's no DAG constraint).

Example: A→B, B→A (cycle). Computing depth of A:
1. Set memo[A] = 1.0
2. Compute depth of B
3. Set memo[B] = 1.0
4. B references A, memo[A] = 1.0, return 1.0
5. depth[B] = 1.0 + 1.0/1 = 2.0, memo[B] = 2.0
6. depth[A] = 1.0 + 2.0/1 = 3.0, memo[A] = 3.0

Now computing depth of B (from a fresh call): memo[B] = 2.0 (stale from last call). But wait, the memo is passed in and reused. If someone queries depth for B after A, they get the cached 2.0, which doesn't account for the final memo[A] = 3.0.

The depth values depend on traversal order, which is non-deterministic for sets.

**Fix:** Either enforce DAG structure (reject cycles at submit time) or use iterative deepening / Tarjan's algorithm for proper cycle-aware depth:

```python
def submit(self, deposit_id, content, references=None, validate=None):
    ...
    # Check for cycles
    if self._would_create_cycle(deposit_id, references or []):
        return False, "Gate 2 REJECTED: references would create a cycle"
    ...
```

---

## HIGH

### H1. `callable` type hint instead of `Callable` / Protocol — bypasses the type system

**File:** `src/othismos/pressure.py:40-42`, `src/othismos/ecology.py:83`

```python
# pressure.py
@dataclass
class Constraint:
    name: str
    type: ConstraintType
    bound_fn: callable        # ← builtin function, not a type
    project_fn: callable      # ← same
    normal_fn: callable | None = None

# ecology.py
def submit(self, ..., validate: callable | None = None) -> tuple[bool, str]:
```

`callable` is a Python builtin *function*, not a type annotation. While it technically works at runtime (Python doesn't enforce annotations), static type checkers (mypy, pyright) treat `callable` as the return type of the `callable()` function, which is incorrect. The library defines proper Protocols in `protocols.py` (`FeasibilityFn`, `ProjectionFn`, `NormalFn`) but doesn't use them.

**Fix:**
```python
from othismos.protocols import FeasibilityFn, ProjectionFn, NormalFn

@dataclass
class Constraint:
    name: str
    type: ConstraintType
    bound_fn: FeasibilityFn
    project_fn: ProjectionFn
    normal_fn: NormalFn | None = None
```

---

### H2. `MoltCycleTracker._all_pressures` grows without bound — memory leak in long training runs

**File:** `src/othismos/phases.py:153`

```python
class MoltCycleTracker:
    def __init__(self, classifier=None):
        ...
        self._all_pressures: list[float] = []
    
    def update(self, pressure: float) -> PhaseReading:
        self._all_pressures.append(pressure)
        reading = self.classifier.classify(self._all_pressures)
```

Unlike `PressureGauge` which has a `window_size`, the tracker accumulates ALL pressures for the entire training run. For a 10M-step training run, this list grows to 10M floats (~80MB just for the list, but the `classify()` method also slices and copies it every call).

The `classify()` method also does `list(pressures[-min(n, window):])` every call, which copies the tail of this ever-growing list.

**Fix:**
```python
class MoltCycleTracker:
    def __init__(self, classifier=None, window_size: int = 10000):
        ...
        self._window_size = window_size
    
    def update(self, pressure: float) -> PhaseReading:
        self._all_pressures.append(pressure)
        if len(self._all_pressures) > self._window_size:
            self._all_pressures = self._all_pressures[-self._window_size:]
        reading = self.classifier.classify(self._all_pressures)
```

---

### H3. `viz.plot_diagnostic_timeline` is O(n²) — will hang on any real training run

**File:** `src/othismos/viz.py:177-190`

```python
for i in range(len(pressures)):
    start = max(0, i - window)
    recent = pressures[start:i + 1]
    result = diag.diagnose(recent, heat=heat)
```

For 10,000 steps, this runs the diagnostic 10,000 times, each with up to 50 data points. Each `diagnose()` call does multiple `np.mean`, `np.std`, `np.polyfit` calls. On a benchmark, this would take minutes to hours for large histories.

**Fix:** Use a rolling/sliding window computation:
```python
# Precompute rolling statistics with pandas or convolution
import pandas as pd
s = pd.Series(pressures)
rolling_mean = s.rolling(window, min_periods=2).mean()
rolling_std = s.rolling(window, min_periods=2).std()
# Classify from precomputed stats
```

---

### H4. `PressureMeasurement.step = -1` sentinel — propagates silently if forgotten

**File:** `src/othismos/pressure.py:114`

```python
return PressureMeasurement(
    step=-1,  # caller can override
    ...
)
```

`compute_othismos` returns a measurement with `step=-1`. `PressureGauge.measure()` overrides it. But any direct call to `compute_othismos` by a user will get `step=-1`, which silently propagates into serialization, analysis, and plotting. There's no warning or validation.

**Fix:** Make step a required parameter, or raise if it's still -1 when serializing:
```python
@dataclass
class PressureMeasurement:
    step: int  # Required, no default
    
    def __post_init__(self):
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
```

---

### H5. `constraint_from_torch_model` passes `type=None` — breaks `ConstraintType` contract

**File:** `src/othismos/integrations.py:227, 233`

```python
constraints.append(
    Constraint(
        name="global_l2",
        type=None,  # type: ignore
        bound_fn=...,
        project_fn=...,
    )
)
```

`Constraint.type` is typed as `ConstraintType` (an enum). Passing `None` with `# type: ignore` means any code that checks `constraint.type` (including `isinstance` checks, serialization, or the `__eq__` on the enum) will break. The `# type: ignore` is an admission of guilt.

**Fix:**
```python
from othismos.pressure import ConstraintType, l2_constraint, box_constraint

def constraint_from_torch_model(model, max_norm=None, max_per_param=None):
    constraints = []
    if max_norm is not None:
        constraints.append(l2_constraint("global_l2", radius=max_norm))
    if max_per_param is not None:
        n_params = sum(p.numel() for p in model.parameters())
        constraints.append(box_constraint(
            "per_param_clip",
            lows=np.full(n_params, -max_per_param),
            highs=np.full(n_params, max_per_param),
        ))
    return constraints
```

---

### H6. `CLI._load_reef` doesn't restore `referenced_by` — orphan checks break after reload

**File:** `src/othismos/cli.py:51-68`

```python
def _load_reef(path: str) -> Reef:
    reef = Reef()
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        for dep_data in data.get("deposits", []):
            reef.submit(dep_data["id"], dep_data["content"],
                        references=dep_data.get("references", []))
```

The JSON stores `referenced_by` explicitly, but `_load_reef` ignores it. Back-references are rebuilt via `submit()` which calls `_compute_depth()`. But the depth scores computed during reload may differ from the original because the order of insertion affects cycle detection in the memo.

Additionally, after reload, `deposit.age` is set directly:
```python
if dep_data.get("age"):
    dep = reef.query(dep_data["id"])
    if dep:
        dep.age = dep_data["age"]
```

But `dep_data.get("age")` returns `0` for new deposits, and `0` is falsy, so deposits with `age=0` don't get their age restored (which is fine since 0 is the default). But if `age` was saved as `0`, this is correct by accident. The real issue: `depth_score` and `layer` are saved but never restored.

**Fix:**
```python
for dep_data in data.get("deposits", []):
    accepted, _ = reef.submit(dep_data["id"], dep_data["content"],
                              references=dep_data.get("references", []))
    if accepted:
        dep = reef.query(dep_data["id"])
        if dep:
            dep.age = dep_data.get("age", 0)
            dep.depth_score = dep_data.get("depth_score", dep.depth_score)
            layer_name = dep_data.get("layer", "SURFACE")
            dep.layer = ReefLayer[layer_name]
```

---

### H7. `_default_kl_distance` doesn't validate inputs — NaN/Inf propagate silently

**File:** `src/othismos/context_pressure.py:69-84`

```python
@staticmethod
def _default_kl_distance(p_full: np.ndarray, p_constrained: np.ndarray) -> float:
    p_full = np.asarray(p_full, dtype=np.float64)
    p_constrained = np.asarray(p_constrained, dtype=np.float64)
    eps = 1e-12
    p_full = p_full + eps
    p_constrained = p_constrained + eps
    p_full = p_full / p_full.sum()
    p_constrained = p_constrained / p_constrained.sum()
    kl_1 = np.sum(p_full * np.log(p_full / p_constrained))
    kl_2 = np.sum(p_constrained * np.log(p_constrained / p_full))
    return float(kl_1 + kl_2)
```

If inputs contain NaN, the entire computation silently produces NaN. If inputs contain negative values (e.g., raw logits, not probabilities), adding eps and normalizing produces garbage — KL divergence is undefined for negative inputs, but this function will happily compute a number.

If inputs are integer arrays (e.g., token IDs), `np.asarray(..., dtype=np.float64)` converts them, but the "probabilities" are just token ID values, producing absurd KL scores.

**Fix:**
```python
@staticmethod
def _default_kl_distance(p_full, p_constrained) -> float:
    p_full = np.asarray(p_full, dtype=np.float64)
    p_constrained = np.asarray(p_constrained, dtype=np.float64)
    
    if p_full.shape != p_constrained.shape:
        raise ValueError(f"Shape mismatch: {p_full.shape} vs {p_constrained.shape}")
    
    # Validate probability distribution
    if np.any(p_full < 0) or np.any(p_constrained < 0):
        raise ValueError("KL divergence requires non-negative inputs")
    
    if np.any(np.isnan(p_full)) or np.any(np.isnan(p_constrained)):
        return float('nan')
    
    eps = 1e-12
    p_full = p_full + eps
    p_constrained = p_constrained + eps
    p_full /= p_full.sum()
    p_constrained /= p_constrained.sum()
    
    return float(np.sum(p_full * np.log(p_full / p_constrained)) +
                 np.sum(p_constrained * np.log(p_constrained / p_full)))
```

---

### H8. `PressureController` never calls `gauge.measure()` — implicit coupling with user

**File:** `src/othismos/controller.py:89-98`

```python
def update(self, current_lr, constraints=None, heat=None):
    ...
    if not self.gauge.history:
        return actions
    pressures = [m.pressure for m in self.gauge.history]
```

The controller reads from `self.gauge.history` but never populates it. The user must call `gauge.measure()` separately before each `controller.update()`. If they forget, the controller returns empty actions with no warning. There's no documentation in the controller's docstring explaining this coupling.

**Fix:** Either document the coupling prominently, or have the controller accept a pressure value:
```python
def update(self, current_lr, pressure: float, constraints=None, heat=None):
    """...
    
    Args:
        pressure: Current pressure reading. Call gauge.measure() to get this.
    """
```

---

### H9. `compute_othismos` doesn't handle NaN/Inf in gradient — produces silent garbage

**File:** `src/othismos/pressure.py:103-130`

If `gradient` contains NaN (common in deep learning with numerical issues), `desired = -lr * gradient` is NaN. All subsequent projections propagate NaN. The `PressureMeasurement` contains NaN arrays and `pressure = NaN`. This NaN then propagates into phase classification, diagnostics, and the controller.

**Fix:**
```python
def compute_othismos(theta, gradient, learning_rate, constraints):
    desired = -learning_rate * gradient
    
    if not np.all(np.isfinite(desired)):
        raise ValueError(
            f"Non-finite values in desired step (gradient has NaN/Inf). "
            f"Check for gradient explosion. LR={learning_rate}"
        )
    ...
```

---

### H10. `PopcornDiagnostic.diagnose` — `cv` computation is unstable for low-pressure regimes

**File:** `src/othismos/diagnostics.py:82`

```python
cv = std_p / mean_p if mean_p > 1e-12 else 0.0
```

When `mean_p` is just above `1e-12` (e.g., `1.1e-12`), `cv` explodes (tiny denominator). A pressure series like `[1e-13, 2e-13, 1e-13]` has `mean_p = 1.33e-13 < 1e-12`, so `cv = 0.0`. But `[1.1e-12, 2.2e-12, 1.1e-12]` has `mean_p = 1.47e-12 > 1e-12`, so `cv = 0.577 / 1.47e-12 ≈ 3.9e11`. This absurd CV value then triggers the Seep classification.

**Fix:** Use a relative threshold or add an absolute floor:
```python
cv = std_p / mean_p if mean_p > 1e-6 else 0.0  # Only compute CV for meaningful pressure
```

Or use a different normalization:
```python
cv = std_p / (mean_p + 1e-8) if mean_p > 1e-12 else 0.0
```

---

## MEDIUM

### M1. `export_metrics_csv` header construction is fragile and buggy

**File:** `src/othismos/serialization.py:95-103`

```python
lines = ["step,pressure," + ",".join(
    sorted({k for m in gauge.history for k in m.pressure_by_constraint})
)]

constraint_keys = [k for k in lines[0].split(",")[2:]]
```

The header is built as a comma-joined string, then re-parsed by splitting on comma. If any constraint name contains a comma, the CSV breaks. The set comprehension is also non-deterministic in ordering (pre-Python 3.7 dict ordering, though CPython 3.7+ preserves insertion order for dicts — but `sorted()` makes it deterministic at least).

**Fix:**
```python
constraint_keys = sorted({k for m in gauge.history for k in m.pressure_by_constraint})
header = ["step", "pressure"] + constraint_keys
lines = [",".join(header)]
```

---

### M2. `OthismosConfig.from_dict` — `__annotations__` check is redundant and may miss inherited fields

**File:** `src/othismos/config.py:46`

```python
valid = {k: v for k, v in data.items() if k in cls.__annotations__ or k in cls.__dataclass_fields__}
```

`cls.__dataclass_fields__` already includes all annotated fields. The `__annotations__` check is redundant. For slotted dataclasses with inheritance, `__dataclass_fields__` is the authoritative source.

**Fix:**
```python
valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
```

---

### M3. API inconsistency: `PressureGauge.measure()` vs `ContextPressureGauge.measure()` parameter order

**File:** `src/othismos/pressure.py:145` vs `src/othismos/context_pressure.py:96`

```python
# PressureGauge
def measure(self, theta, gradient, learning_rate, constraints):
    
# ContextPressureGauge  
def measure(self, full_output, constrained_output, context_tokens_dropped=0, **metadata):
```

The two gauge classes have completely different `measure()` signatures. This isn't inherently wrong (they measure different things), but a user switching between them will need to relearn the API. Consider a common base class or protocol.

---

### M4. `Reef.tick()` iterates over deposits while deleting — safe via `list()` copy but still surprising

**File:** `src/othismos/ecology.py:131-150`

```python
for dep_id, deposit in list(self._deposits.items()):
    ...
    if deposit.is_orphan and deposit.age > self.erosion_age:
        eroded.append(dep_id)
        del self._deposits[dep_id]
```

The `list()` copy prevents `RuntimeError: dictionary changed size during iteration`, but deleting from `_deposits` while the copy's `deposit` object is still being used is fine here since we only read from it. However, `fail_deposit()` during a tick (if someone subclasses) would break.

---

### M5. `PhaseClassifier` is stateful across calls — `_last_high_pressure_step` leaks between sequences

**File:** `src/othismos/phases.py:73`

```python
class PhaseClassifier:
    def __init__(self, ...):
        ...
        self._last_high_pressure_step: int | None = None
```

The classifier stores `_last_high_pressure_step` as instance state. If the same classifier is used to classify different pressure sequences (e.g., in a multi-worker setup or unit tests), the state from one sequence leaks into the next. The classifier has no `reset()` method.

**Fix:**
```python
def reset(self) -> None:
    """Reset classifier state for a new sequence."""
    self._last_high_pressure_step = None
```

---

### M6. `Reef.search` is O(n) string scan — fine for small reefs, terrible for large ones

**File:** `src/othismos/ecology.py:169-175`

```python
def search(self, query: str, limit: int = 10) -> list[Deposit]:
    results = []
    query_lower = query.lower()
    for deposit in self._deposits.values():
        if query_lower in deposit.content.lower():
            results.append(deposit)
    results.sort(key=lambda d: d.depth_score, reverse=True)
    return results[:limit]
```

For 100K deposits, this scans every deposit's content on every search call. No index, no caching.

---

### M7. `box_constraint` uses `ConstraintType.CUSTOM` — loses type information

**File:** `src/othismos/pressure.py:105`

```python
def box_constraint(name: str, lows: np.ndarray, highs: np.ndarray) -> Constraint:
    return Constraint(
        name=name,
        type=ConstraintType.CUSTOM,  # Should be ConstraintType.BOX?
        ...
    )
```

The `ConstraintType` enum has `L2_NORM`, `CONTEXT`, `THERMAL`, `COMPUTE`, `MEMORY`, `LATENCY`, `CUSTOM`. There's no `BOX`. This means box constraints are indistinguishable from any other custom constraint.

**Fix:** Add `BOX = "box"` to the enum, or document that box constraints should use `CUSTOM`.

---

### M8. `OthismosConfig` `lr_bounds` is a tuple — immutable but not validated

**File:** `src/othismos/config.py:28`

```python
lr_bounds: tuple[float, float] = (1e-6, 1.0)
```

No validation that `lr_bounds[0] <= lr_bounds[1]` or that both are positive. A user could set `lr_bounds=(1.0, 1e-6)` and the controller would silently fail to ever adjust the LR (since `max(min_lr, lr * factor)` would always clamp to `min_lr` which is > `max_lr`).

**Fix:**
```python
def __post_init__(self):
    if self.lr_bounds[0] > self.lr_bounds[1]:
        raise ValueError(f"lr_bounds[0] must be <= lr_bounds[1], got {self.lr_bounds}")
    if self.lr_bounds[0] < 0:
        raise ValueError(f"lr_bounds must be non-negative, got {self.lr_bounds}")
```

---

## LOW

### L1. `PressureMeasurement.is_pushing` threshold `1e-10` is arbitrary and undocumented

**File:** `src/othismos/pressure.py:60`

```python
@property
def is_pushing(self) -> bool:
    return self.pressure > 1e-10
```

Why 1e-10? This should be configurable or at least documented.

---

### L2. `MoltPhase` IntEnum values are not documented as an ordering contract

**File:** `src/othismos/phases.py:14-20`

The `IntEnum` values 0-4 are ordered by "intensity" but this contract isn't documented. Code that relies on `MoltPhase.CRISIS > MoltPhase.RESISTANCE` should have this documented explicitly.

---

### L3. `OthismosTrainerCallback.on_pre_optimizer_step` ignores non-model args silently

**File:** `src/othismos/integrations.py:272`

```python
def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
    if model is not None:
        self._inner.pre_step(model, None, None)
```

If `model` is `None` (which happens in some HF Trainer configurations), the callback silently does nothing. No warning.

---

### L4. `token_overlap` docstring says "Jaccard distance" but function name says `token_overlap`

**File:** `src/othismos/context_pressure.py:135`

Minor naming inconsistency — the function returns Jaccard distance but is named `token_overlap`, which suggests the overlap coefficient (a different metric).

---

### L5. `PressureGauge.pressure_trend` hardcodes window of 50

**File:** `src/othismos/pressure.py:183`

```python
n = min(len(self._history), 50)
```

The trend window should be configurable or at least documented as a constant.

---

## Cross-Cutting Concerns

### Thread Safety

None of the classes are thread-safe. `PressureGauge`, `MoltCycleTracker`, `Reef`, and `PressureController` all mutate internal state without locks. In multi-worker training (DDP, FSDP), using these from multiple threads will cause data corruption. This should at least be documented.

### Private Attribute Access

Multiple modules access private attributes (`_history`, `_step`, `_deposits`, `_all_pressures`, `_current_cycle`) across class boundaries:
- `integrations.py` manipulates `gauge._history` directly
- `serialization.py` reads `gauge._step`
- `cli.py` accesses `reef._deposits`
- `pandas_export.py` accesses `reef._deposits` and `tracker._current_cycle`

This makes refactoring dangerous — changing a "private" attribute name will break dependent modules.

### Missing `__all__` in Modules

Individual modules don't define `__all__`. Only `__init__.py` does. Users who import from specific modules (e.g., `from othismos.pressure import _l2_ball_project`) will get unintended access to private functions.

---

## Verdict

The library has a strong conceptual foundation and the code is readable. But it has real bugs that will bite users: the `or`-vs-`is None` threshold bug (C6), the wrong pressure computation in the Torch callback (C3), the memory blowup on real models (C4), and the order-dependent constraint application (C2) are the most dangerous. The `callable` type hint (H1) and `type=None` with `# type: ignore` (H5) show awareness of type issues that were deferred rather than fixed.

**Priority fixes for v0.4.0:**
1. C6: `or` → `is None` (1-line fix per occurrence)
2. C3: Fix `OthismosTorchCallback` pressure computation
3. C4: Document model-size limitation or add chunked computation
4. C2: Document the sequential projection limitation
5. H1: Replace `callable` with Protocol types from `protocols.py`
6. C1: Fix `l2_constraint` default center
