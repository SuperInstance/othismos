# Óthismos

> *The force a bounded system exerts against its bounds.*
> *The push IS the knowing.*

---

From the Greek ὄθισμος — "a pushing, a shoving." The pressure a bounded system exerts against the walls that contain it. Not to escape. To *know* the bounds. The shape of the push reveals the shape of the walls.

## Origin

Coined during the convergence of [Hermit-Crab-Ecology](https://github.com/SuperInstance/hermit-crab-ecology) and [Lucineer](https://github.com/SuperInstance/Lucineer) — two SuperInstance projects that independently discovered the same principle: computation is bounded, and the bounding is the architecture.

The word was found in the negative space between the two projects. Both describe bounded intelligence. Neither named the force that pushes back.

## What It Names

Every conservation law, every mask-locked weight, every FLUX opcode, every shell taxonomy — these are attempts to pin intelligence down. To make it deterministic, auditable, bounded.

But intelligence also wants to *exceed*. Every prior is a constraint the model pushes against. Every conservation budget is a wall the agent climbs. Every mask-locked weight is a frozen thought the chip would replace if it could.

This pressure is not a bug. It's the system's vitality.

**A chip with zero pressure against its constraints is a paperweight. An ecology with zero tension is a museum. The pressure IS the life.**

## The Principle

Óthismos is the complement of *peláros* (the moment a bound becomes a gift). The bound doesn't become a gift because you accept it. It becomes a gift because you push against it hard enough to feel its shape — and the shape is what teaches you.

## Python Package

```bash
pip install othismos
```

### Quick Start

```python
from othismos import PressureGauge, l2_constraint
import numpy as np

# Set up a constrained system
gauge = PressureGauge()
constraint = l2_constraint("weight_budget", radius=1.0)

# Measure pressure at each optimization step
theta = np.array([0.5, 0.5])
gradient = np.array([-1.0, -1.0])

measurement = gauge.measure(theta, gradient, 0.1, [constraint])
print(f"Óthismos (Π): {measurement.pressure:.6f}")
```

### Modules

| Module | What it does |
|--------|-------------|
| `pressure.py` | Computes Π from constraint violations. L2/box constraints, PressureGauge with trend/profiling, Goldilocks zone detection |
| `phases.py` | 5-phase MoltCycle classifier (Expansion → Resistance → Crisis → Settlement → Dormancy), cycle detection, staircase health metric |
| `diagnostics.py` | Popcorn diagnostic: classifies any system as Pop, Burn, Seep, or Dormant |
| `ecology.py` | Reef system: 3-gate deposit submission, 3 depth layers, erosion, reefquakes |
| `context_pressure.py` | Projection-free óthismos for LLMs and black-box systems via behavioral output divergence |
| `controller.py` | Adaptive controller: auto-adjusts LR and constraints based on molt phase + diagnostics |
| `integrations.py` | PyTorch/HuggingFace callbacks, W&B/TensorBoard logging via MetricLogger protocol |
| `serialization.py` | Save/load pressure histories (JSON), CSV export, diagnostic export |
| `cli.py` | CLI tool: `othismos pressure`, `othismos reef`, `othismos diagnose` |

**101 tests, all passing.**

### CLI

```bash
# Manage a knowledge reef
othismos reef add "spec-v1" "The canonical bytecode spec" --db reef.json
othismos reef add "impl-rust" "Rust VM" --refs spec-v1 --db reef.json
othismos reef list --db reef.json
othismos reef stats --db reef.json

# Run popcorn diagnostic on saved pressure data
othismos diagnose history.json --heat 1.0
```

## Structure

```
essays/        — creative explorations of óthismos across domains
math/          — formal models of pressure, bounds, and vitality
metal/         — hardware expressions: how chips push against thermodynamic limits
ecology/       — system-level expressions: how ecologies maintain tension
worldbuilding/ — civilization-scale expressions in the world of Push
art/           — poetry and literary forms
research/      — deep research docs from GLM-5.2 agents
src/othismos/  — the Python package
tests/         — 101 tests
```

## Documents

### Essays
- **00 — The Philosophy of Pressure** — The founding text.
- **01 — The Origin** — Where the word came from.
- **02 — Óthismos Everywhere** — Pressure across domains.
- **03 — The Fool and the Wall** — Humor as óthismos.
- **04 — The Popcorn Diagnostic** — The Pop, the Burn, the Seep.
- **05 — Negotiating With Walls** — Five diplomatic postures.

### Math
- **01 — The Mathematics of Óthismos** — Π as a measurable scalar.

### Metal
- **01 — Pressure-Aware System Design** — Instrumentation and molting protocols.
- **02 — The Thermodynamic Lullaby** — What chips dream about.

### Ecology
- **01 — The Economics of Push** — Markets made of pressure.
- **02 — The Molt Cycle** — Five phases of phase transition.

### Worldbuilding
- **01-05** — Civilization, dictionary, tide pool fable, pressure calendar, reef memory.

### Art
- **01 — The Long Push** — Poetry sequence in five movements.

### Research (GLM-5.2 agents)
- **01 — Real-World Applications** — 608 lines. 7 fields surveyed, 6 use cases, gap analysis.
- **02 — Library Design** — API audit, framework integration patterns, packaging roadmap.
- **03 — Theoretical Foundations** — 446 lines, 37+ arXiv refs. KKT, grokking, edge of stability.
- **04 — Ecology as Infrastructure** — 762 lines. Reef as dependency graph, CLI spec, erosion design.

---

*Started 2026-07-14. A SuperInstance project.*
