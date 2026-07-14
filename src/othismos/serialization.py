"""
Serialization for óthismos measurements and histories.

Save/load pressure histories, phase readings, and diagnostic results
as JSON for later analysis or comparison across runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from othismos.pressure import PressureMeasurement, PressureGauge, GoldilocksZone
from othismos.phases import PhaseReading, MoltPhase
from othismos.diagnostics import DiagnosticResult, SystemHealth


def _ndarray_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays and types to JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    elif isinstance(obj, set):
        return sorted(_ndarray_to_list(v) for v in obj)
    return obj


def save_history(gauge: PressureGauge, path: str | Path) -> None:
    """Save a PressureGauge history to JSON."""
    def _ndarray_to_list(obj: Any) -> Any:
        """Recursively convert numpy arrays and types to JSON-serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, dict):
            return {str(k): _ndarray_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_ndarray_to_list(v) for v in obj]
        elif isinstance(obj, set):
            return sorted(_ndarray_to_list(v) for v in obj)
        return obj

    data = {
        "version": "0.1.0",
        "step_count": gauge._step,
        "measurements": [
            {
                "step": _ndarray_to_list(m.step),
                "pressure": _ndarray_to_list(m.pressure),
                "pressure_by_constraint": _ndarray_to_list(m.pressure_by_constraint),
                "desired_step": _ndarray_to_list(m.desired_step),
                "actual_step": _ndarray_to_list(m.actual_step),
                "violation": _ndarray_to_list(m.violation),
            }
            for m in gauge.history
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_history(path: str | Path) -> PressureGauge:
    """Load a PressureGauge from a JSON file."""
    data = json.loads(Path(path).read_text())
    gauge = PressureGauge(window_size=max(len(data["measurements"]) * 2, 1000))
    gauge._step = data.get("step_count", 0)

    for m_data in data["measurements"]:
        m = PressureMeasurement(
            step=m_data["step"],
            desired_step=np.array(m_data["desired_step"]),
            actual_step=np.array(m_data["actual_step"]),
            violation=np.array(m_data["violation"]),
            pressure=m_data["pressure"],
            pressure_by_constraint=m_data.get("pressure_by_constraint", {}),
        )
        gauge._history.append(m)

    return gauge


def save_diagnostic(result: DiagnosticResult, path: str | Path) -> None:
    """Save a diagnostic result to JSON."""
    data = {
        "health": result.health.value,
        "pressure": result.pressure,
        "heat": result.heat,
        "pressure_efficiency": result.pressure_efficiency,
        "leak_rate": result.leak_rate,
        "confidence": result.confidence,
        "recommendation": result.recommendation,
        "signals": result.signals,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def export_metrics_csv(gauge: PressureGauge, path: str | Path) -> None:
    """Export pressure history as CSV for external analysis."""
    lines = ["step,pressure," + ",".join(
        sorted({k for m in gauge.history for k in m.pressure_by_constraint})
    )]

    constraint_keys = [k for k in lines[0].split(",")[2:]]

    for m in gauge.history:
        row = [str(m.step), f"{m.pressure:.8f}"]
        for key in constraint_keys:
            row.append(f"{m.pressure_by_constraint.get(key, 0.0):.8f}")
        lines.append(",".join(row))

    Path(path).write_text("\n".join(lines))


def pressure_summary(gauge: PressureGauge) -> dict[str, Any]:
    """Get a JSON-serializable summary of a pressure gauge."""
    history = gauge.history
    if not history:
        return {"status": "empty", "step_count": 0}

    pressures = [m.pressure for m in history]
    zone = gauge.goldilocks()

    return {
        "status": "active",
        "step_count": len(history),
        "current_pressure": gauge.current_pressure,
        "mean_pressure": gauge.mean_pressure,
        "max_pressure": max(pressures),
        "min_pressure": min(pressures),
        "pressure_trend": gauge.pressure_trend,
        "goldilocks_zone": {
            "lower": zone.lower_bound,
            "upper": zone.upper_bound,
            "width": zone.width,
        },
        "constraint_profile": gauge.pressure_profile(),
    }
