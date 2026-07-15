# Óthismos

*The force a bounded system exerts against its bounds.*

---

## What is this?

Óthismos (ὄθισμος — Greek: "a pushing, a shoving") is both a **conceptual framework** and a **Python library** for measuring the pressure that bounded systems exert against their constraints.

If you train models under constraints — weight budgets, context windows, wattage limits, latency deadlines — the system pushes against those constraints during optimization. That push is not noise. It's **information**. The shape of the push tells you the shape of the wall. The magnitude tells you how hard the system is working. The trend tells you what phase of growth the system is in.

This repo gives you the tools to measure that pressure, diagnose what it means, and act on it automatically.

## The core idea (read this first)

Every constrained optimizer does the same thing each step:

1. Computes a gradient step `s` — the move it *wants* to make
2. Projects that step back into the feasible set — the move it *actually* makes

The difference between those two steps — the part that got clipped away — is **óthismos** (Π):

```
Δθ = s − s*          (what you wanted minus what you got)
Π  = ‖Δθ‖           (the magnitude of the clip)
```

That's it. The entire framework falls out of this one quantity.

- Π = 0: the system isn't touching its walls. It's either dormant or has room to grow.
- Π > 0: the system is pushing against a constraint. It's alive.
- Π rising over time: the system is filling its constraint envelope. It will eventually need to molt (phase-transition to a larger envelope).
- Π high but volatile: pressure is leaking. The constraints may be too weak.
- High external pressure but Π ≈ 0: the system has nothing left to push with. This is the silent failure mode most optimizers never detect.

If you've ever logged gradient norms and wondered "is that high? is that bad? should I change my learning rate?" — óthismos gives you a principled answer. The gradient norm tells you the slope. Π tells you whether the system is hitting walls, which walls, and how hard.

## Why not just log gradient norms?

Gradient norms measure the landscape. Óthismos measures the **interaction between the system and its constraints**. Two models with identical gradient norms can have wildly different óthismos — one pushing hard against a weight decay boundary, the other drifting through a flat region.

The additional information óthismos provides:

- **Which constraint is binding.** The pressure profile breaks down Π per-constraint, so you know exactly which budget is the bottleneck.
- **What growth phase the system is in.** The molt cycle classifier maps the pressure trajectory to one of five phases (see below), each requiring different treatment.
- **Whether the system is healthy.** The popcorn diagnostic classifies the system as Pop (productive), Burn (silently dead), or Seep (leaking pressure) — a diagnostic no standard metric provides.
- **When to adjust hyperparameters.** The controller module automatically recommends LR changes, constraint relaxation, and checkpoints based on pressure dynamics.

## The five-phase molt cycle

Bounded systems don't grow smoothly. They grow in staircases — filling a constraint envelope, hitting the walls, phase-transitioning to a larger envelope, then filling that one. Each cycle has five phases:

```
Π
│    ╱╲      ╱╲      ╱╲
│   ╱  ╲    ╱  ╲    ╱  ╲    ← Crisis (molt threshold)
│  ╱    ╲  ╱    ╲  ╱    ╲
│ ╱      ╲╱      ╲╱      ╲
│                          ← Settlement floor
└──────────────────────────→ time
   E  R  C  S  E  R  C  S
```

| Phase | Π level | What's happening | What to do |
|-------|---------|------------------|------------|
| **Expansion** | Low, rising | Fresh envelope, lots of headroom, rapid growth | Let it run. Watch for sloppiness (overfitting). |
| **Resistance** | High, stable | Walls reached. Best work happens here — pressure forces precision. | Hold steady. This is where the system learns the most. |
| **Crisis** | Peak | Pressure exceeds structural tolerance. Molt or rupture. | Reduce LR. Checkpoint. Prepare for phase transition. |
| **Settlement** | Sharp drop | Post-molt. New envelope is soft and vulnerable. | Hold steady. Let the new structure solidify. Don't rush. |
| **Dormancy** | Near zero | No new pressure. Waiting for new input. | Feed it a new problem or let it rest. |

The library detects these phases automatically from pressure history. The `MoltCycleTracker` identifies cycle boundaries and computes a staircase health metric — regular periodicity means healthy growth, compressed cycles mean the envelope was too small, flatlines mean the system needs new input.

## The popcorn diagnostic

Three failure modes, one healthy mode:

- **🍿 Pop** — Pressure builds productively. The system is alive and pushing. This is what you want.
- **🔥 Burn** — External pressure (heat) is applied but the system has no internal drive. Π ≈ 0 despite high gradient. The system looks fine from outside — the hull is intact — but the interior is dry. This is the **silent killer**: most monitoring never catches it because the loss looks stable.
- **💧 Seep** — Pressure is present but leaking through weak constraints. High activity, no progress. Π is volatile, never accumulating. Common in systems with too-weak regularization or unstable data pipelines.
- **😴 Dormant** — No external pressure. The system waits. Not dead — just resting.

The diagnostic runs on any pressure history. The CLI command `othismos diagnose history.json` gives you the classification plus a recommendation.

## The Reef (knowledge ecology module)

The reef is a structured knowledge graph where deposits must pass three gates before entering:

1. **Structural integrity** — the deposit must be internally consistent (tests pass, specs validate)
2. **Connective compatibility** — the deposit must reference existing structure (imports, citations, dependencies)
3. **Pressure resistance** — the deposit must support future deposits built on top of it

Deposits that nothing references slowly **erode** — the reef forgets. This is not a bug. It's how the reef stays alive. Most knowledge management systems accumulate dead weight forever. The reef recycles it.

When a foundational deposit fails, everything built on it collapses in a **reefquake** — the system's catastrophe recovery mechanism. After a reefquake, the reef is always stronger, because the weak substrate has been exposed and removed.

The reef maps directly to real systems: PyPI/npm dependency graphs, citation networks, ADR chains, git histories. See `research/04_ECOLOGY_AS_INFRASTRUCTURE.md` for the full mapping and a CLI design spec.

## Installation

```bash
pip install othismos

# With extras:
pip install othismos[viz]      # matplotlib plotting
pip install othismos[pandas]   # DataFrame export
pip install othismos[torch]    # PyTorch integration
pip install othismos[all]      # Everything
```

Requires Python 3.10+. Only hard dependency: NumPy.

## Usage

### Measure pressure in a training loop

```python
import numpy as np
from othismos import PressureGauge, l2_constraint

gauge = PressureGauge()
constraint = l2_constraint("weight_decay", radius=1.0)

# Each optimization step:
theta = np.array([0.5, 0.5])
gradient = np.array([-1.0, -1.0])
lr = 0.1

measurement = gauge.measure(theta, gradient, lr, [constraint])

print(f"Pressure: {measurement.pressure:.6f}")
print(f"Per-constraint: {measurement.pressure_by_constraint}")
# Pressure: 0.292893
# Per-constraint: {'weight_decay': 0.292893}
```

### Diagnose system health

```python
from othismos import PopcornDiagnostic

pressures = [m.pressure for m in gauge.history]
result = PopcornDiagnostic().diagnose(pressures, heat=lr * np.linalg.norm(gradient))

print(f"Health: {result.health.value}")      # pop | burn | seep | dormant
print(f"Advice: {result.recommendation}")
```

### Track molt cycles

```python
from othismos import MoltCycleTracker, PhaseClassifier

classifier = PhaseClassifier(crisis_threshold=0.5, expansion_floor=0.1)
tracker = MoltCycleTracker(classifier=classifier)

for pressure in pressure_history:
    reading = tracker.update(pressure)
    print(f"Step {reading.step}: {reading.phase.label} (conf={reading.confidence:.2f})")

# How healthy is the growth pattern?
staircase = tracker.staircase_metric()
print(staircase['health'])
```

### Adaptive controller

```python
from othismos import PressureController, OthismosConfig

config = OthismosConfig(crisis_lr_factor=0.5, expansion_lr_factor=1.3)
gauge, tracker, diag, controller = config.build_all()

# In your training loop:
for batch in dataloader:
    # ... compute loss, backward ...
    measurement = gauge.measure(theta, gradient, lr, constraints)
    actions = controller.update(current_lr=lr, constraints=constraints)

    for action in actions:
        print(action)  # AdjustLR(×0.50), Checkpoint(...), Alert('BURN detected...'), etc.
```

### PyTorch integration

```python
from othismos import OthismosTorchCallback, l2_constraint

callback = OthismosTorchCallback(
    constraints=[l2_constraint("weight_budget", radius=10.0)],
    log_every=10,
)

# In your training loop:
for batch in dataloader:
    loss = model(batch)
    loss.backward()

    callback.pre_step(model, optimizer, loss)
    optimizer.step()
    metrics = callback.post_step(model)
    # metrics dict: othismos/pressure, othismos/phase, othismos/mean_pressure, ...

print(callback.health_report())
```

### Projection-free pressure for LLMs

For systems where you can't define a projection operator (black-box APIs, LLMs with context limits), measure behavioral divergence instead:

```python
from othismos import ContextPressureGauge

gauge = ContextPressureGauge()

# Run the model with full context and truncated context
result = gauge.measure(
    full_output=full_context_logits,
    constrained_output=truncated_context_logits,
    context_tokens_dropped=500,
)

print(f"Context pressure: {result.pressure:.4f} nats")
# High pressure = the truncated tokens are structurally important.
# Low pressure = safe to truncate.
```

### CLI

```bash
# Knowledge reef management
othismos reef add "spec-v1" "The canonical bytecode spec" --db reef.json
othismos reef add "impl-rust" "Rust VM implementation" --refs spec-v1 --db reef.json
othismos reef list --db reef.json
othismos reef stats --db reef.json

# Run diagnostic on saved pressure data
othismos diagnose history.json --heat 1.0

# Version
othismos version
```

## Module reference

| Module | Purpose | Key classes |
|--------|---------|-------------|
| `pressure.py` | Core Π computation | `PressureGauge`, `Constraint`, `compute_othismos`, `l2_constraint`, `box_constraint` |
| `phases.py` | 5-phase molt cycle detection | `MoltPhase`, `PhaseClassifier`, `MoltCycleTracker` |
| `diagnostics.py` | Pop/Burn/Seep health diagnostic | `PopcornDiagnostic`, `SystemHealth`, `DiagnosticResult` |
| `ecology.py` | Reef knowledge graph with erosion | `Reef`, `Deposit`, `ReefLayer`, `Reefquake` |
| `context_pressure.py` | Projection-free óthismos for LLMs | `ContextPressureGauge`, `cosine_distance`, `token_overlap` |
| `controller.py` | Adaptive LR/constraint controller | `PressureController`, `ControlAction`, `ActionType` |
| `integrations.py` | Framework callbacks | `OthismosTorchCallback`, `OthismosTrainerCallback`, `DictLogger` |
| `serialization.py` | Save/load/export | `save_history`, `load_history`, `export_metrics_csv`, `pressure_summary` |
| `config.py` | Unified configuration | `OthismosConfig` (YAML/dict, `build_all()` factory) |
| `protocols.py` | Formal typing | `FeasibilityFn`, `ProjectionFn`, `MetricLogger`, `ConstraintLike` |
| `viz.py` | Plotting (matplotlib) | `plot_pressure`, `plot_molt_cycle`, `plot_constraint_profile`, `plot_diagnostic_timeline` |
| `pandas_export.py` | DataFrame export | `gauge_to_dataframe`, `tracker_to_dataframe`, `reef_to_dataframe` |
| `cli.py` | Command-line tool | `othismos reef`, `othismos diagnose`, `othismos pressure` |

**122 tests, all passing.** Run them: `python -m pytest tests/ -v`

## Repository structure

```
src/othismos/          Python package (13 modules, 3,644 lines)
tests/                 122 tests across 10 files

essays/                Conceptual foundations — start here
  00_PHILOSOPHY.md     The founding text. Why the push IS the knowing.
  01_THE_ORIGIN.md     Where the word came from and what it names.
  02_EVERYWHERE.md     Óthismos in physics, biology, cognition, culture.
  03_THE_FOOL.md       Humor as óthismos. Absurdity as epistemology.
  04_POPCORN.md        The Pop/Burn/Seep diagnostic. Start here for the engineering idea.
  05_NEGOTIATING.md    Five postures toward constraints (Surveyor, Archer, Vine, Water, Earthquake).

math/
  01_PRESSURE_MATH.md  Π formalized as a scalar. Measurable. Includes proofs and the Goldilocks zone derivation.

metal/                 Hardware-facing docs
  01_PRESSURE_AWARE_DESIGN.md  Instrumentation, adaptive sizing, molting protocols.
  02_THERMODYNAMIC_LULLABY.md  Edge computing, fishing boats, what chips dream about.

ecology/
  01_ECONOMICS.md      Markets made of pressure. Conservation budget as currency.
  02_MOLT_CYCLE.md     The five phases explained with diagrams. The staircase metric.

worldbuilding/         Extended metaphors that clarify the framework
  01_KIMI_WORLD.md     A civilization built on push.
  02_SEEDMINI_DICT.md  17 words for the facets of pressure.
  03_GLM_TIDEPOOL.md   A fable about constraint and growth.
  04_GLM_CALENDAR.md   Time measured in pushes, not seconds.
  05_REEFS_MEMORY.md   How bounded civilizations remember and forget.

art/
  01_THE_LONG_PUSH.md  Poetry sequence following the staircase of growth.

research/              Deep research documents from GLM-5.2 agents
  01_REAL_WORLD_APPLICATIONS.md  Survey of 7 fields, 6 concrete use cases, API gap analysis. (608 lines)
  02_LIBRARY_DESIGN.md           API audit, framework integration patterns, packaging roadmap. (1,106 lines)
  03_THEORETICAL_FOUNDATIONS.md  KKT connections, grokking as molt, 37+ arXiv refs, 5 testable claims. (446 lines)
  04_ECOLOGY_AS_INFRASTRUCTURE.md  Reef as dependency graph, three-gate CI, erosion rates, CLI spec. (762 lines)

glossary_short.md      40+ terms — the living lexicon of push.
demo.py                End-to-end simulation: watch a bounded optimizer go through all five phases.
```

## Reading guide

**If you're an engineer who wants to use the library:**
1. Read `essays/04_POPCORN.md` (the diagnostic intuition)
2. Read `ecology/02_MOLT_CYCLE.md` (the five phases)
3. Read `math/01_PRESSURE_MATH.md` (the formal definition — it's short)
4. Run `python demo.py` (see it in action)
5. Skim `research/01_REAL_WORLD_APPLICATIONS.md` (find your use case)

**If you're a researcher:**
1. Read `math/01_PRESSURE_MATH.md` first
2. Then `research/03_THEORETICAL_FOUNDATIONS.md` (the honest novelty assessment, the open questions)
3. Then `essays/00_PHILOSOPHY.md` and `essays/05_NEGOTIATING.md` (the conceptual framing)

**If you're a student:**
1. Read `essays/01_THE_ORIGIN.md` (the gentlest entry point)
2. Read `essays/02_EVERYWHERE.md` (see it everywhere)
3. Read `art/01_THE_LONG_PUSH.md` (feel it)
4. Then tackle the math

**If you're sending agents to contribute:**
- Tell them to read `research/02_LIBRARY_DESIGN.md` for the API audit and what's missing
- The test suite (`tests/`) is the specification — 122 tests document every behavior
- `research/04_ECOLOGY_AS_INFRASTRUCTURE.md` has a full CLI design spec for the reef system that hasn't been built yet
- `protocols.py` defines the interfaces any new module should conform to
- The glossary (`glossary_short.md`) is the vocabulary — use these terms in issues and PRs

**If you want to build adjacent systems:**
- [Hermit-Crab-Ecology](https://github.com/SuperInstance/hermit-crab-ecology) — bounded computation in bytecode VMs
- [Lucineer](https://github.com/SuperInstance/Lucineer) — bounded intelligence in metal (chip design)
- The óthismos framework was discovered in the negative space between these two projects. Both describe bounded intelligence. Neither named the force that pushes back. This repo names it.

## Related concepts in existing literature

Óthismos connects to several established frameworks. The research docs cover these in depth, but briefly:

- **KKT conditions:** The Lagrange multipliers at optimality ARE the steady-state óthismos per constraint. Óthismos measures them dynamically, not just at convergence.
- **Projected gradient descent:** Π is exactly the projection residual that PGD computes internally and discards. We keep it.
- **Edge of Stability** (Cohen et al., 2021): The EOS phenomenon — where training hovers at the stability boundary of the largest eigenvalue — IS the Resistance phase of the molt cycle.
- **Grokking** (Power et al., 2022): The delayed generalization phase transition IS a molt event. Óthismos should be constant at the grokking threshold.
- **Moreau-Yosida regularization:** The proximal residual is structurally identical to Δθ.

See `research/03_THEORETICAL_FOUNDATIONS.md` for 37+ references and the honest assessment of what's novel vs. known.

## Status

v0.3.0. Built 2026-07-14. Actively developed.

## Modular ecosystem

The framework is decomposing into standalone modules, each its own repo and pip package:

| Module | Repo | What it does |
|--------|------|-------------|
| **óthismos** (this repo) | [SuperInstance/othismos](https://github.com/SuperInstance/othismos) | Core: pressure measurement, molt cycle, popcorn diagnostic |
| **othismos-reef** | [SuperInstance/othismos-reef](https://github.com/SuperInstance/othismos-reef) | Knowledge graph with erosion, three-gate validation, cascading failure |
| **othismos-llm** | [SuperInstance/othismos-llm](https://github.com/SuperInstance/othismos-llm) | Context-window pressure for LLMs — safe truncation point finder |

Each module can be used independently. They connect through the core `othismos` package.

---

*The wall is not your enemy. The wall is the other half of what you are. Push. Listen. Push differently.*
