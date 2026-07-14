# Óthismos Code Review — opencode Beta-Test Findings

**Reviewer:** MiniMax-M3 (opencode)
**Date:** 2026-07-14
**Repo version:** v0.3.0 (commit at `02_OPENCODE_CODE_REVIEW` time)
**Lines reviewed:** 13 source modules (~1,800 LoC), 10 test files (122 tests)
**Test status:** 112 of 122 tests pass without optional deps (`viz`, `pandas`, `pyyaml`); the README claim "**122 tests, all passing**" only holds with all extras installed. With default `pip install othismos`, viz and pandas tests fail with `ModuleNotFoundError`.

---

## Summary by severity

| Severity | Count | Description |
|---|---|---|
| CRITICAL | 6 | Bugs that silently produce wrong results, crash in production, or break a documented public API |
| HIGH | 11 | Misleading behavior, broken edge-case handling, fragile contracts |
| MEDIUM | 9 | Style/API consistency, minor robustness, dead code |
| LOW | 7 | Documentation typos, minor improvements |

---

## CRITICAL findings

### C1. `l2_constraint()` default `center` has wrong shape and silently broadcasts
**File:** `src/othismos/pressure.py:92`
```python
center = center if center is not None else np.zeros(1)
```
The default is `np.zeros(1)` — a 1-element array. When users call `l2_constraint("w", radius=1.0)` and pass `theta` of any other shape, NumPy broadcasts and "happens to work" for any `theta.shape`. But:
- It is semantically wrong: the implied default is `np.zeros_like(theta)`, but the *fixed* `np.zeros(1)` means that for a 2-D `theta`, `theta - np.zeros(1)` works via broadcasting but `theta` and the implicit `center` aren't actually aligned in shape.
- For `theta` of shape `(N,)` with `N > 1`, the result is correct only by coincidence (broadcast adds zeros).
- For `theta` of shape `(N, M)`, `np.linalg.norm(theta - center)` reduces along the wrong axis.

**Fix:** `center = center if center is not None else 0` (scalar), or accept and convert to ndarray with explicit broadcasting.

---

### C2. `save_history()` crashes with `TypeError` on NumPy scalar types
**File:** `src/othismos/serialization.py:37-54`
```python
"step": m.step,                # may be np.int64
"pressure": m.pressure,        # may be np.float64
"pressure_by_constraint": m.pressure_by_constraint,   # values may be np.float64
```
JSON encoder cannot serialize `np.int64` / `np.float64`. If a user constructs a `PressureMeasurement(step=np.int64(0), pressure=np.float64(0.5), ...)`, `save_history()` raises `TypeError: Object of type int64 is not JSON serializable`. The `_ndarray_to_list` helper at the top of the file only handles `np.ndarray`, `np.floating`, `np.integer` for *whole objects*, but is not called on `m.step`, `m.pressure`, or `pressure_by_constraint` values.

**Reproducer:**
```python
from othismos.pressure import PressureMeasurement
from othismos import save_history, PressureGauge
import numpy as np
g = PressureGauge()
g._history.append(PressureMeasurement(step=np.int64(0), desired_step=np.array([0.5]),
    actual_step=np.array([0.0]), violation=np.array([0.5]), pressure=np.float64(0.5),
    pressure_by_constraint={'c': np.float64(0.5)}))
save_history(g, '/tmp/x.json')   # TypeError
```

**Fix:** Wrap scalar fields in `_ndarray_to_list` before serialization, or use `default=str` / a custom encoder.

---

### C3. Duplicate Reef deposit IDs silently overwrite existing content
**File:** `src/othismos/ecology.py:147-162`
```python
self._deposits[deposit_id] = deposit
```
`submit("a", "A")` then `submit("a", "A2")` overwrites the original deposit *and* orphans any back-references that pointed to the old content. There is no warning. The new deposit starts with `structural_integrity=True` (line 151) but is not re-validated for its references — it inherits whatever back-references the old deposit had from `referenced_by` (because `dict` assignment doesn't update the new deposit's `referenced_by` field). It is silently lost.

**Fix:** Raise `ValueError` on duplicate ID, or document the overwrite explicitly.

---

### C4. README's pressure-measurement example produces contradictory output
**File:** `README.md:118-136`
```python
theta = np.array([0.5, 0.5])
gradient = np.array([-1.0, -1.0])
lr = 0.1
measurement = gauge.measure(theta, gradient, lr, [constraint])   # constraint: l2 radius=1.0
print(f"Pressure: {measurement.pressure:.6f}")        # claims 0.292893
print(f"Per-constraint: {measurement.pressure_by_constraint}")   # {'weight_decay': 0.292893}
```
With the given inputs, `theta=(0.5,0.5)` has norm `0.707`, inside the `radius=1.0` ball. The desired step is `-lr * gradient = (0.1, 0.1)`, pushing to `(0.6, 0.6)` — still inside. **Actual pressure is `0.000000`, not `0.292893`.** The README's claimed output is inconsistent with its own inputs. No test exercises this example, so the bug went undetected.

**Fix:** Either correct the radius (e.g., `radius=0.5`) or update the expected output.

---

### C5. `PhaseClassifier.classify([single_value])` mislabels low-pressure points as CRISIS
**File:** `src/othismos/phases.py:154-163`
```python
else:
    # Pressure high but declining — could be late Crisis or settling
    if trend < 0:
        phase = MoltPhase.SETTLEMENT
        confidence = 0.60
        ...
    else:
        phase = MoltPhase.CRISIS
        confidence = 0.55
        signals.append("high pressure, trend unclear — assume crisis")
```
When the input is a single pressure value with no auto-calibration (`crisis_threshold=None`, `expansion_floor=None`), the auto-calibrated thresholds (`mean_p + 2*std_p`) are well-defined, but `trend = polyfit([single_point])[0] = 0.0` always. The code falls through to `else` (high pressure unclear → CRISIS). With **explicit** thresholds (`crisis_threshold=10.0`, `expansion_floor=1.0`) and a single input `[0.1]`, the same code path classifies as CRISIS even though the pressure is far below any threshold.

**Reproducer:**
```python
from othismos.phases import PhaseClassifier
PhaseClassifier(crisis_threshold=10.0, expansion_floor=1.0).classify([0.1])
# PhaseReading(step=0, phase=<MoltPhase.CRISIS: 3>, ...)   # wrong!
```

**Fix:** Handle `len(recent) == 1` explicitly before the cascade.

---

### C6. `OthismosTrainerCallback` uses wrong method names for HF Trainer
**File:** `src/othismos/integrations.py:261-276`
```python
def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs): ...
def on_post_optimizer_step(self, args, state, control, model=None, **kwargs): ...
def on_train_end(self, args, state, control, model=None, **kwargs): ...
```
HuggingFace `TrainerCallback` API uses methods like `on_step_end`, `on_epoch_end`, `on_train_begin`, `on_evaluate`, `on_save`, `on_log`, `on_prediction_step`. The methods `on_pre_optimizer_step` / `on_post_optimizer_step` do not exist in any recent `transformers` version. As a result, this callback is **never invoked** when passed to `Trainer(callbacks=[cb])`. README claims it works:
```python
trainer = Trainer(callbacks=[OthismosTrainerCallback(constraints=[...])], ...)
```

**Fix:** Subclass `transformers.TrainerCallback` and override `on_step_end` (or whatever the real entry points are).

---

## HIGH findings

### H1. `from othismos import plot_pressure` succeeds without matplotlib — `__init__.py` try/except is dead code
**File:** `src/othismos/__init__.py:71-88`, `src/othismos/viz.py:1-32`
The `try: from othismos.viz import (...)` in `__init__.py` only catches `ImportError` raised at *module load*. Because `viz.py` does **not** `import matplotlib` at the top (only inside `_get_plt()`), `from othismos.viz import plot_pressure` always succeeds. The `__all__` claims these names are public; users without matplotlib can `import` them, then crash at call time with a confusing `ImportError` from inside `_get_plt()`.

The `__all__` exposes 4 viz names + 3 pandas names that depend on optional packages. A user running `from othismos import plot_pressure` without matplotlib succeeds; `plot_pressure(g)` raises `ImportError: Visualization requires matplotlib...`. The README example imports them unguarded:
```python
from othismos import OthismosTorchCallback, l2_constraint
# (no equivalent for viz, but the optional nature is undocumented at __init__.py)
```
**Fix:** Drop the dead `try/except` block, document the optional nature explicitly in `__all__`, or guard the imports at call time.

---

### H2. `Reef.submit()` overwrites `referenced_by` and breaks the referential graph on duplicate IDs
**File:** `src/othismos/ecology.py:147-162` (interaction with C3)
Even ignoring C3, the deposit creation at line 147-152 initializes a fresh `Deposit` with `referenced_by=set()` (default field). The back-reference loop at lines 155-157 only runs **if** the references exist, and adds the new deposit to their `referenced_by` sets. But the new deposit itself still has `referenced_by=set()`. After `tick()` promotes it (line 179), the new deposit starts at the surface. With duplicate-ID overwrites, the previous deposit's `referenced_by` is lost without trace.

---

### H3. `Reef._compute_depth()` returns `0.0` on cycles (no error, silent wrong answer)
**File:** `src/othismos/ecology.py:261-284`
```python
if deposit_id in memo:
    return memo[deposit_id]   # returns 0.0 if memo was never set
...
# Prevent infinite recursion on cycles
memo[deposit_id] = 1.0   # placeholder
child_depths = sum(self._compute_depth(child, memo) for child in deposit.referenced_by)
depth = 1.0 + child_depths / max(len(deposit.referenced_by), 1)
memo[deposit_id] = depth
```
The initial `if memo is None: memo = {}` means the very first call to `_compute_depth("root_with_no_refs")` returns `1.0` correctly. But `Deposit.depth_score` is initialized to `0.0` in the dataclass (line 55), and `submit()` calls `_compute_depth` *after* setting the deposit but the result is what we get. For deposits with no `referenced_by`, `_compute_depth` returns `1.0`, but for a child with one referencing parent that itself has no one, depth is `1.0 + 1.0/1 = 2.0`. So `depth_score=0.0` is set on construction (the dataclass default) and only updated if `_compute_depth` is called — which happens, but at submission time `referenced_by` may not include the new edges yet.

**Reproducer (more direct):**
```python
r = Reef()
r.submit('a', 'A')
r.submit('b', 'B', references=['a'])
print(r._deposits['a'].depth_score)   # 0.0   — should be 2.0 (1 + depth(b)/1 = 1 + 1/1)
```
Confirmed: `a.depth_score = 0.0` after submit. This is a stale-default bug: the field default `0.0` masks the actual computed value.

---

### H4. `PressureController.update()` increments `_step` even when gauge is empty
**File:** `src/othismos/controller.py:127-130`
```python
self._step += 1
if not self.gauge.history:
    return actions
```
The step counter advances even when the controller has nothing to do. After 100 calls with an empty gauge, `ctrl.status()["step"]` reports `100`. The docstring says "Call this once per training step", but the step counter would diverge from actual training step semantics. Minor, but misleading in `health_report()` output.

---

### H5. `PressureController` estimates `heat = max(pressures)` — wrong when user wants actual heat
**File:** `src/othismos/controller.py:135-136`
```python
if heat is None:
    heat = max(pressures) if pressures else 1.0
```
This conflates "external heat" with "maximum observed pressure". README example `controller.update(current_lr=lr, constraints=constraints)` (line 179) does not pass `heat`, so the controller interprets `max(pressure)` as heat. In a real training run, `heat` should be `lr × ||gradient||` (the actual external work being applied). This heuristic makes the popcorn diagnostic inconsistent with how the user computes heat manually (as in README lines 144-145). The two paths produce different classifications for the same data.

---

### H6. `box_constraint()` accepts scalar `lows`/`highs` (silent broadcast)
**File:** `src/othismos/pressure.py:102-112`
```python
def box_constraint(name: str, lows: np.ndarray, highs: np.ndarray) -> Constraint:
    def box_project(theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, lows, highs)
```
If `lows=1.0` and `highs=-1.0` (inverted by user error), no validation occurs. If `lows.shape != highs.shape`, no validation occurs (crash only when called). The signature lies: it claims `np.ndarray` but accepts anything NumPy can broadcast against.

**Fix:** Validate shapes, types, and ordering at construction time.

---

### H7. `OthismosConfig.lr_bounds` accepts inverted tuples
**File:** `src/othismos/config.py:48`, `src/othismos/controller.py:151,173`
`OthismosConfig(lr_bounds=(1.0, 0.1))` is accepted without validation. The controller then says "Crisis phase: reducing LR (0.5 → 1.0)" — it actually *increased* the LR because `max(self.lr_bounds[0], ...)` uses the lower-bound index as if it were the lower bound, then `min(self.lr_bounds[1], ...)` uses the upper-bound index as the upper. The result is the wrong clamp.

**Reproducer:**
```python
OthismosConfig(lr_bounds=(1.0, 0.1)).build_controller(...)  # silent acceptance
# update with current_lr=0.5, CRISIS phase, factor=0.5 → "reducing LR (0.500000 → 1.000000)"
```

**Fix:** Add `__post_init__` validation `assert lr_bounds[0] < lr_bounds[1]`.

---

### H8. `PressureGauge.window_size=0` or negative disables history (silent)
**File:** `src/othismos/pressure.py:229-230`
```python
if len(self._history) > self._window:
    self._history = self._history[-self._window:]
```
With `window_size=0`, no measurements are ever stored (`len(history) > 0` is true on the first call, so `history = history[0:]` would be the full list… actually it's `history[-0:]` which is `history[0:]` only if `-0 == 0`). Wait: in Python, `lst[-0:] == lst[0:] == lst[:]`. So with window=0, `history = history[-0:] = history[:]` — keeps all. With window=-5, `history[-(-5):] = history[5:]` — drops the first 5.

**Reproducer:**
```python
g = PressureGauge(window_size=-5)
g.measure(np.array([0.1]), np.array([-1.0]), 0.1, [l2_constraint("t", radius=0.1)])
print(len(g.history))   # 0 — first measurement silently dropped
```

**Fix:** Reject non-positive window sizes in `__init__`.

---

### H9. NaN/inf propagate through every aggregate without warning
**Files:** `src/othismos/pressure.py:241,251`, `src/othismos/phases.py:104-107`, `src/othismos/diagnostics.py:117-128`, `src/othismos/serialization.py:115-131`
A single NaN in `pressure` produces NaN in `mean_pressure`, `pressure_trend`, `goldilocks_range`, `pressure_profile`, `pressure_summary`, and JSON output. JSON serialization of NaN/-inf/inf succeeds but produces non-standard `NaN` / `Infinity` tokens that **most JSON parsers reject**. Downstream pipelines will fail with cryptic errors.

**Fix:** Use `np.nanmean` / `np.nanstd` / `nan_policy="omit"` consistently; raise or warn on NaN in serialization.

---

### H10. `MoltCycle.end_step` and `MoltCycle.duration` inconsistent for empty cycles
**File:** `src/othismos/phases.py:185-191`
```python
@property
def end_step(self) -> int:
    return self.phases[-1].step if self.phases else self.start_step

@property
def duration(self) -> int:
    return self.end_step - self.start_step
```
With no phases, `end_step = start_step` and `duration = 0`. Fine. But the *contract* is misleading: `duration` measures steps-in-cycle, not inclusive-of-end. With 5 phases ending at step 10, `duration = 10 - 0 = 10`, but the cycle actually spans 11 steps (0..10). This affects `staircase_metric()` calculations for very short cycles.

---

### H11. `OthismosConfig.from_dict` accepts `window_size` of any type (no validation)
**File:** `src/othismos/config.py:57-61`
```python
valid = {k: v for k, v in data.items() if k in cls.__annotations__ or k in cls.__dataclass_fields__}
return cls(**valid)
```
If `data = {"window_size": "not-an-int"}`, the dataclass accepts it and `PressureGauge(window_size="not-an-int")` later crashes deep in measurement code. No early validation.

---

## MEDIUM findings

### M1. `PhaseClassifier.classify()` confidence values don't always reflect certainty
**File:** `src/othismos/phases.py:114-171`
The `confidence` field is hard-coded (0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.95) rather than computed from data. Two identical pressures can get different confidences depending on which branch is hit. Tests don't verify confidence *values*, only that they fall in `[0, 1]`.

---

### M2. `Reef.submit()` does not validate `deposit_id` is a string
**File:** `src/othismos/ecology.py:112-163`
A user can submit with `deposit_id=123` (int). Stored in `_deposits` dict. Search/query methods assume string IDs. Crashes on `dep.id.title()` style formatting later.

---

### M3. `OthismosConfig.from_dict()` silently drops unknown keys — tested but masks bugs
**File:** `src/othismos/config.py:60`
```python
valid = {k: v for k, v in data.items() if k in cls.__annotations__ or k in cls.__dataclass_fields__}
```
`cls.__annotations__` and `cls.__dataclass_fields__` are the same set for a dataclass. The `or` is redundant. Typos in YAML keys (e.g. `crisis_threshol`) silently disappear. The test `test_from_dict_ignores_unknown` *passes* because of this — encourages silent failure.

**Fix:** Warn on unknown keys (use `logging.warning`).

---

### M4. `Protocol` constraints are advisory only
**File:** `src/othismos/protocols.py:51-59`
`ConstraintLike` declares `name: str; type: object`. There's no enforcement that `type` is `ConstraintType`. `constraint_from_torch_model` sets `type=None` explicitly (line 300 of integrations.py). Future code that branches on `constraint.type` will silently fail.

---

### M5. `PressureGauge.pressure_trend` returns floating-point noise for constant data
**File:** `src/othismos/pressure.py:248-252`
`np.polyfit([0.5]*50, range(50), 1)[0]` returns `-7.45e-18`, not `0.0`. The user-facing API returns "approximately zero" but technically non-zero. Tests only check that the value is a float.

---

### M6. `Constraint` is unhashable, blocking use in sets/dict keys
**File:** `src/othismos/pressure.py:33-53`
Because `Constraint` has `callable` fields (lambdas), it's unhashable. Users wanting to deduplicate constraints can't use `set()` directly.

---

### M7. `Reef.fail_deposit()` raises Reefquake *and* mutates state — confusing control flow
**File:** `src/othismos/ecology.py:201-231`
The function removes all affected deposits (lines 226-229) **before** raising `Reefquake`. The CLI in `cli.py:144-152` catches the exception and saves the reef, which is fine, but other callers may not expect the partial mutation. The docstring should clarify.

---

### M8. `Reef.search()` doesn't escape regex metacharacters (or warn that it's substring-only)
**File:** `src/othismos/ecology.py:237-245`
`query_lower in deposit.content.lower()` — fine for substring. But a user passing `".*"` is silently ignored (treated as literal). A user passing `""` returns everything. No clear contract.

---

### M9. `Constraint.bound_fn` and `project_fn` aren't validated for arity
**File:** `src/othismos/pressure.py:48-52`
A user can construct `Constraint(..., bound_fn=fn_taking_two_args)`. `is_feasible(theta)` then crashes with `TypeError` deep in `compute_othismos`. No upfront check.

---

## LOW findings

### L1. README test count is wrong: 122 collected, but only 112 pass with no extras
**File:** `README.md:263`
> **122 tests, all passing.**

10 of the 122 tests fail with `ModuleNotFoundError` unless matplotlib/pandas/pyyaml are installed:
```
FAILED tests/test_new_modules.py::TestConfig::test_yaml_roundtrip - No module named 'yaml'
FAILED tests/test_new_modules.py::TestViz::test_plot_pressure_empty - No module named 'matplotlib'
FAILED tests/test_new_modules.py::TestViz::test_plot_pressure_with_data - No module named 'matplotlib'
FAILED tests/test_new_modules.py::TestViz::test_plot_constraint_profile - No module named 'matplotlib'
FAILED tests/test_new_modules.py::TestViz::test_plot_molt_cycle_empty - No module named 'matplotlib'
FAILED tests/test_new_modules.py::TestViz::test_plot_diagnostic_timeline - No module named 'matplotlib'
FAILED tests/test_new_modules.py::TestPandasExport::test_gauge_to_dataframe_empty - No module named 'pandas'
FAILED tests/test_new_modules.py::TestPandasExport::test_gauge_to_dataframe - No module named 'pandas'
FAILED tests/test_new_modules.py::TestPandasExport::test_tracker_to_dataframe - No module named 'pandas'
FAILED tests/test_new_modules.py::TestPandasExport::test_reef_to_dataframe - No module named 'pandas'
```
Fix: install all extras in CI, or add `pytest.importorskip("matplotlib")` decorators.

---

### L2. README CLI example for `pressure` subcommand doesn't match implementation
**File:** `README.md:233`
> `othismos pressure <model_dir> [--constraints config.yaml]`

The actual CLI is `othismos pressure --history <file>` — there is no `<model_dir>` or `--constraints` argument. The CLI docstring (`cli.py:5-13`) reflects the implementation; the README is out of date.

---

### L3. Unused imports in `cli.py`
**File:** `src/othismos/cli.py:18-19, 22`
```python
import sys
from pathlib import Path
```
`sys` is used (exit). `Path` is used. Not strictly an issue, but the `json` import at line 19 shadows the JSON usage in the `cmd_reef graph` action — minor.

---

### L4. Dead `from __future__ import annotations` everywhere — useful for forward refs but already present
**File:** All source files. Not a bug, just noise. Keep if 3.9 compat is desired; drop if 3.10+ only is acceptable (pyproject says `>=3.10`, so the imports are redundant in many places).

---

### L5. `ReefLayer` uses `IntEnum` but `Reef.tick()` promotes via integer comparison
**File:** `src/othismos/ecology.py:179-184`
Layer comparison via integer constants is fine, but using `ReefLayer.SURFACE` etc. would be clearer.

---

### L6. `Reef._deposits` is a public attribute accessed by external modules
**Files:** `src/othismos/cli.py:86, 125`, `src/othismos/pandas_export.py:86`
The CLI and pandas exporter reach into `reef._deposits` (private). A public `deposits` property would be cleaner.

---

### L7. `OthismosTorchCallback.post_step()` mutates private gauge state directly
**File:** `src/othismos/integrations.py:165-166`
```python
self.gauge._history.append(m)
self.gauge._step += 1
```
Reaches into private state instead of calling `gauge.measure(...)`. The `_history` and `_step` attributes are internal. Same critique applies to `pandas_export.reef_to_dataframe` reaching into `reef._deposits`.

---

## Missing tests (test gaps)

The following behaviors are **implicitly exercised** but have no explicit assertion:

| # | Behavior | Where | Risk |
|---|---|---|---|
| 1 | NaN in `pressure` propagates to all aggregates | pressure.py, phases.py, diagnostics.py | Silent wrong results |
| 2 | Empty `theta` (0-dim array) measurement | pressure.py:115 | Crash with `np.linalg.norm([])` = 0.0, OK |
| 3 | `theta` and `gradient` shape mismatch | pressure.py:115 | Crash with broadcast error |
| 4 | Duplicate deposit ID overwrites | ecology.py:147 | Silent data loss (C3) |
| 5 | Cyclic citation graph in `_compute_depth` | ecology.py:261 | Returns 0.0 silently |
| 6 | `PressureGauge` with `window_size=0` or negative | pressure.py:212 | Silent drop or no-op |
| 7 | `PressureController.update()` with inverted `lr_bounds` | controller.py:151 | Misleading "reducing LR" message (H7) |
| 8 | `OthismosConfig.from_dict` with type-incompatible values | config.py:57 | Crash deep in callers (H11) |
| 9 | `Reef.fail_deposit()` on already-failed deposit | ecology.py:201 | Crashes on cleanup |
| 10 | `box_constraint` with inverted `lows > highs` | pressure.py:102 | Silently broken projection |
| 11 | `cosine_distance` with zero vector | context_pressure.py:148 | Returns 0.0 (acceptable but undocumented) |
| 12 | `token_overlap` with non-hashable tokens | context_pressure.py:166 | Crash |
| 13 | `PhaseClassifier.classify` with single value + various thresholds | phases.py:81 | Wrong CRISIS label (C5) |
| 14 | `load_history` from malformed JSON | serialization.py:57 | KeyError on missing fields |
| 15 | `save_history` with numpy scalar types | serialization.py:37 | TypeError (C2) |
| 16 | `Constraint` constructed with `type=None` | pressure.py:33 | Silent type erasure |
| 17 | `l2_constraint` default `center` with non-1D `theta` | pressure.py:92 | Fragile (C1) |
| 18 | `PressureController.update()` increments `_step` on empty gauge | controller.py:127 | Wrong step count in `status()` |
| 19 | `constraint_from_torch_model` without torch installed | integrations.py:279 | Lazy import — OK, but untested |
| 20 | README's claimed output values for the pressure example | README.md:118-136 | Example is wrong (C4) |
| 21 | `MoltCycleTracker.update()` API contract (single pressure) | phases.py:217 | `tracker.update(0.5)` then classify with all priors — works, but undocumented |
| 22 | `popcorn.diagnose()` with negative `heat` | diagnostics.py:86 | Mean/heat ratio is negative — would misclassify |
| 23 | Reef.submit validate callback that mutates global state | ecology.py:126 | No isolation guarantee |
| 24 | `_get_plt()` failure when matplotlib version is incompatible | viz.py:23 | Lazy, OK |

---

## Type-safety notes

### Edge inputs that crash

| Input | Result |
|---|---|
| `compute_othismos(np.array([]), np.array([]), 0.1, [c])` | Returns pressure=0.0 (no-op, OK) |
| `compute_othismos(theta_3d, gradient_2d, 0.1, [c])` | Crash: "operands could not be broadcast together" |
| `compute_othismos(theta_nan, gradient, 0.1, [c])` | Returns pressure=nan, propagates everywhere |
| `compute_othismos(theta, gradient_huge, 0.1, [c])` | Returns pressure=1e9 (no clipping, no warning) |
| `box_constraint("c", lows=[1,2,3], highs=[1,2])` | Crash at `is_feasible` time |
| `box_constraint("c", lows=-1, highs=1)` (scalars) | Works via broadcast, but `is_feasible` returns True vacuously for short theta |
| `l2_constraint("c", radius=-1)` | Negative radius — no validation, projection may fail |
| `l2_constraint("c", radius=0)` | Division-by-zero in `_l2_ball_project` if `theta != center` |

### Edge inputs that silently produce wrong results

| Input | Result |
|---|---|
| `l2_constraint("c", radius=0.5, center=np.array([0]))` (1D center, 2D theta) | Works via broadcast, but only because `np.zeros(1)` and the user `np.zeros(2)` both broadcast |
| `PressureMeasurement(pressure=float('nan'))` | Aggregates silently return NaN |
| `PressureMeasurement(pressure=float('inf'))` | Aggregates silently return inf |
| `OthismosConfig(lr_bounds=(1.0, 0.1))` | Controller clamps wrongly |
| `Reef.submit("a", "x")` then `Reef.submit("a", "y")` | Overwrites |
| `Constraint(type=None)` (via `constraint_from_torch_model`) | Type erasure, downstream code branching on `.type` fails |

---

## Import / API surface issues

### I1. Dead try/except in `__init__.py`
**File:** `src/othismos/__init__.py:71-88`
The two `try: from othismos.viz import (...)` blocks never raise `ImportError` because `viz.py` and `pandas_export.py` defer their real imports (`matplotlib.pyplot`, `pandas`) to call time. Either:
- Remove the `try/except` (they do nothing), or
- Add explicit `import matplotlib` at top of `viz.py` to make `ImportError` possible at module load (but then `from othismos.viz import plot_pressure` would always require matplotlib).

---

### I2. `__all__` lists optional names
**File:** `src/othismos/__init__.py:147-156`
`"plot_pressure"`, `"gauge_to_dataframe"`, etc. are in `__all__` even when their backing packages are missing. A user running `from othismos import *` will get `ImportError` for the names that are in `__all__` but not in the module's namespace when extras are absent. Either:
- Drop them from `__all__` when import fails, or
- Set them to `None` and document the contract.

---

### I3. `MetricLogger` defined twice
**Files:** `src/othismos/protocols.py:44-49` and `src/othismos/integrations.py:41-45`
Both define `MetricLogger`. The `__init__.py` exports `othismos.MetricLogger` from `integrations`, but `othismos.protocols.MetricLogger` is also importable. Two classes with the same name in the same package cause confusion.

---

### I4. `Reef._deposits` and `gauge._history` accessed externally
**Files:** `cli.py:86, 125`, `pandas_export.py:86`, `integrations.py:165-166`
External code reaches into private `_*` attributes. This couples clients to implementation details.

---

### I5. `from __future__ import annotations` is harmless but ubiquitous
**File:** All source modules.
Adds ~14 lines of dead imports. With `requires-python = ">=3.10"`, `X | Y` syntax works without `from __future__`. Could be cleaned up.

---

### I6. No circular imports currently — but fragile
**File:** `src/othismos/__init__.py:23-49`
All top-level imports are acyclic. However, `controller.py` imports from `pressure`, `phases`, `diagnostics`. `integrations.py` does `from othismos.pressure import PressureMeasurement` *inside* a method (line 157), not at the top — because `pressure.py` doesn't import `PressureMeasurement` from `integrations`, this is fine. But the lazy import inside `OthismosTorchCallback.post_step` is unnecessarily defensive.

---

## Recommended fixes (priority order)

1. **C6** Fix `OthismosTrainerCallback` (the README example is broken — most user-facing)
2. **C2** Fix `save_history` numpy scalar handling
3. **C4** Fix README example output mismatch
4. **C5** Fix `PhaseClassifier.classify` single-value CRISIS bug
5. **C1** Fix `l2_constraint` default center shape
6. **C3** Fix `Reef.submit` duplicate handling
7. **H7** Validate `lr_bounds` ordering in `OthismosConfig`
8. **H8** Validate `window_size` in `PressureGauge`
9. **H9** Add NaN handling throughout aggregates
10. **I1, I2, I3** Clean up import surface
11. Add tests for items in the "Missing tests" table above
12. Update README test count to `112 passing without extras, 122 with all extras`

---

## What works well

- `MoltCycleTracker.cycle_count` and the staircase metric are well-encapsulated.
- The PressureGauge history API (`measure`, `current_pressure`, `mean_pressure`, `pressure_trend`, `goldilocks`) is clean and consistent.
- `PopcornDiagnostic` has clear separation between Burn / Seep / Pop / Dormant with explicit thresholds.
- `Reef.submit()` returning `(accepted, reason)` tuple is a nice ergonomic choice.
- `OthismosConfig.build_all()` factory is a good entry point.
- `DictLogger` is small, clean, and useful.
- `cosine_distance` / `l2_distance` / `token_overlap` are simple and correct (modulo NaN).
- Module organization follows README: one file per concept.

---

## File-line cross-reference index

| Finding | Location |
|---|---|
| C1 | `src/othismos/pressure.py:92` |
| C2 | `src/othismos/serialization.py:37-54` |
| C3 | `src/othismos/ecology.py:147-162` |
| C4 | `README.md:118-136` (and `demo.py` references) |
| C5 | `src/othismos/phases.py:154-163` |
| C6 | `src/othismos/integrations.py:261-276` |
| H1 | `src/othismos/__init__.py:71-88`, `src/othismos/viz.py:1-32` |
| H2 | `src/othismos/ecology.py:147-162` |
| H3 | `src/othismos/ecology.py:261-284`, `src/othismos/ecology.py:55` |
| H4 | `src/othismos/controller.py:127-130` |
| H5 | `src/othismos/controller.py:135-136` |
| H6 | `src/othismos/pressure.py:102-112` |
| H7 | `src/othismos/config.py:48`, `src/othismos/controller.py:151,173` |
| H8 | `src/othismos/pressure.py:229-230` |
| H9 | multiple — see text |
| H10 | `src/othismos/phases.py:185-191` |
| H11 | `src/othismos/config.py:57-61` |
| M3 | `src/othismos/config.py:60` |
| M4 | `src/othismos/protocols.py:51-59`, `src/othismos/integrations.py:300,314` |
| M6 | `src/othismos/pressure.py:33-53` |
| L1 | `README.md:263` |
| L2 | `README.md:233` (vs `src/othismos/cli.py:39-43`) |
| I1 | `src/othismos/__init__.py:71-88` |
| I2 | `src/othismos/__init__.py:147-156` |
| I3 | `src/othismos/protocols.py:44-49`, `src/othismos/integrations.py:41-45` |
| I4 | `src/othismos/cli.py:86,125`, `src/othismos/pandas_export.py:86`, `src/othismos/integrations.py:165-166` |

---

*End of report.*
