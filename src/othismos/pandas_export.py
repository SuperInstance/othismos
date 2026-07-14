"""
Pandas export utilities for óthismos.

Convert pressure histories and reef data to DataFrames for analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from othismos.pressure import PressureGauge
    from othismos.phases import MoltCycleTracker
    from othismos.ecology import Reef


def gauge_to_dataframe(gauge: "PressureGauge"):
    """Convert a PressureGauge history to a pandas DataFrame.

    Returns columns: step, pressure, and one column per constraint.
    """
    import pandas as pd

    history = gauge.history
    if not history:
        return pd.DataFrame()

    # Collect all constraint names
    all_constraints = set()
    for m in history:
        all_constraints.update(m.pressure_by_constraint.keys())
    all_constraints = sorted(all_constraints)

    rows = []
    for m in history:
        row = {"step": m.step, "pressure": m.pressure}
        for c in all_constraints:
            row[f"constraint_{c}"] = m.pressure_by_constraint.get(c, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def tracker_to_dataframe(tracker: "MoltCycleTracker"):
    """Convert a MoltCycleTracker history to a pandas DataFrame.

    Returns columns: step, phase, pressure, confidence, cycle_number.
    """
    import pandas as pd

    rows = []
    for cycle in tracker.cycles:
        for r in cycle.phases:
            rows.append({
                "step": r.step,
                "phase": r.phase.name,
                "phase_id": int(r.phase),
                "pressure": r.pressure,
                "confidence": r.confidence,
                "cycle_number": cycle.cycle_number,
            })

    # Add current cycle if in progress
    if tracker._current_cycle:
        for r in tracker._current_cycle.phases:
            rows.append({
                "step": r.step,
                "phase": r.phase.name,
                "phase_id": int(r.phase),
                "pressure": r.pressure,
                "confidence": r.confidence,
                "cycle_number": tracker._current_cycle.cycle_number,
            })

    return pd.DataFrame(rows)


def reef_to_dataframe(reef: "Reef"):
    """Convert a Reef's deposits to a pandas DataFrame.

    Returns columns: id, layer, age, reference_count, depth_score, content_preview.
    """
    import pandas as pd

    rows = []
    for dep in reef._deposits.values():
        rows.append({
            "id": dep.id,
            "layer": dep.layer.name,
            "age": dep.age,
            "reference_count": dep.reference_count,
            "depth_score": dep.depth_score,
            "is_orphan": dep.is_orphan,
            "content_length": len(dep.content),
            "content_preview": dep.content[:80],
        })

    return pd.DataFrame(rows).sort_values("depth_score", ascending=False).reset_index(drop=True)
