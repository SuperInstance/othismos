"""
Visualization helpers for óthismos.

Plot pressure histories, molt cycles, and diagnostic states.
Uses matplotlib (optional dependency).

Usage:
    >>> from othismos.viz import plot_pressure, plot_molt_cycle
    >>> plot_pressure(gauge)  # shows or returns Figure
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from othismos.pressure import PressureGauge
    from othismos.phases import MoltCycleTracker


def _get_plt():
    """Import matplotlib lazily."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install othismos[viz]"
        )


_PHASE_COLORS = {
    0: "#2c3e50",  # Dormancy - dark
    1: "#27ae60",  # Expansion - green
    2: "#f39c12",  # Resistance - orange
    3: "#e74c3c",  # Crisis - red
    4: "#3498db",  # Settlement - blue
}

_PHASE_NAMES = {
    0: "Dormancy",
    1: "Expansion",
    2: "Resistance",
    3: "Crisis",
    4: "Settlement",
}


def plot_pressure(
    gauge: "PressureGauge",
    show_goldilocks: bool = True,
    show_trend: bool = True,
    ax=None,
    **kwargs,
):
    """Plot pressure history over time.

    Args:
        gauge: PressureGauge with history
        show_goldilocks: Shade the Goldilocks zone
        show_trend: Overlay a trend line
        ax: Optional matplotlib axis
        **kwargs: Passed to plot()

    Returns:
        matplotlib Figure
    """
    plt = _get_plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    history = gauge.history
    if not history:
        ax.text(0.5, 0.5, "No pressure data", ha="center", va="center", transform=ax.transAxes)
        return fig

    steps = [m.step for m in history]
    pressures = [m.pressure for m in history]

    # Goldilocks zone
    if show_goldilocks and len(history) >= 10:
        zone = gauge.goldilocks()
        ax.axhspan(zone.lower_bound, zone.upper_bound, alpha=0.15, color="green", label="Goldilocks zone")

    # Pressure line
    ax.plot(steps, pressures, linewidth=0.8, alpha=0.8, label="Π (óthismos)", **kwargs)

    # Trend line
    if show_trend and len(pressures) > 10:
        z = np.polyfit(steps, pressures, 1)
        trend = np.polyval(z, steps)
        ax.plot(steps, trend, "--", color="gray", alpha=0.6, linewidth=1, label=f"Trend (slope={z[0]:.6f})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Pressure (Π)")
    ax.set_title("Óthismos — Pressure History")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if fig:
        fig.tight_layout()
    return fig


def plot_molt_cycle(
    tracker: "MoltCycleTracker",
    ax=None,
):
    """Plot the molt cycle staircase.

    Shows pressure over time colored by phase, with cycle boundaries marked.

    Args:
        tracker: MoltCycleTracker with history
        ax: Optional matplotlib axis

    Returns:
        matplotlib Figure
    """
    plt = _get_plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    if not tracker.cycles and not tracker._current_cycle:
        ax.text(0.5, 0.5, "No cycle data", ha="center", va="center", transform=ax.transAxes)
        return fig

    all_cycles = list(tracker.cycles)
    if tracker._current_cycle and tracker._current_cycle.phases:
        all_cycles.append(tracker._current_cycle)

    for i, cycle in enumerate(all_cycles):
        if not cycle.phases:
            continue

        steps = [r.step for r in cycle.phases]
        pressures = [r.pressure for r in cycle.phases]
        phases = [int(r.phase) for r in cycle.phases]

        # Color points by phase
        scatter = ax.scatter(steps, pressures, c=[_PHASE_COLORS.get(p, "#999") for p in phases],
                           s=8, alpha=0.6, zorder=2)

        # Draw line
        ax.plot(steps, pressures, linewidth=0.5, alpha=0.4, color="#555", zorder=1)

        # Mark cycle boundary
        if i > 0:
            ax.axvline(x=cycle.start_step, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    # Legend for phases
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_PHASE_COLORS[i], label=_PHASE_NAMES[i])
        for i in range(5)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Pressure (Π)")
    ax.set_title(f"Molt Cycle Staircase — {len(all_cycles)} cycles detected")
    ax.grid(True, alpha=0.2)

    if fig:
        fig.tight_layout()
    return fig


def plot_constraint_profile(
    gauge: "PressureGauge",
    ax=None,
):
    """Plot per-constraint pressure breakdown as a stacked bar chart.

    Args:
        gauge: PressureGauge with history
        ax: Optional matplotlib axis

    Returns:
        matplotlib Figure
    """
    plt = _get_plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    profile = gauge.pressure_profile()
    if not profile:
        ax.text(0.5, 0.5, "No constraint data", ha="center", va="center", transform=ax.transAxes)
        return fig

    names = list(profile.keys())
    values = list(profile.values())

    bars = ax.barh(names, values, color="#3498db", alpha=0.7)
    ax.set_xlabel("Mean Pressure (Π)")
    ax.set_title("Per-Constraint Pressure Profile")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.6f}", va="center", fontsize=8)

    if fig:
        fig.tight_layout()
    return fig


def plot_diagnostic_timeline(
    gauge: "PressureGauge",
    heat: float = 1.0,
    window: int = 50,
    ax=None,
):
    """Plot pressure with diagnostic health overlaid.

    Colors the background by health state (Pop=green, Burn=red, Seep=yellow).

    Args:
        gauge: PressureGauge with history
        heat: External heat for diagnostic
        window: Diagnostic window size
        ax: Optional matplotlib axis

    Returns:
        matplotlib Figure
    """
    plt = _get_plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    history = gauge.history
    if not history:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    from othismos.diagnostics import PopcornDiagnostic, SystemHealth

    diag = PopcornDiagnostic()
    steps = [m.step for m in history]
    pressures = [m.pressure for m in history]

    # Run diagnostic at each point
    health_colors = []
    for i in range(len(pressures)):
        start = max(0, i - window)
        recent = pressures[start:i + 1]
        if len(recent) < 2:
            health_colors.append("#bdc3c7")  # gray for insufficient data
            continue
        result = diag.diagnose(recent, heat=heat)
        if result.health == SystemHealth.POP:
            health_colors.append("#27ae60")
        elif result.health == SystemHealth.BURN:
            health_colors.append("#e74c3c")
        elif result.health == SystemHealth.SEEP:
            health_colors.append("#f1c40f")
        else:
            health_colors.append("#bdc3c7")

    ax.scatter(steps, pressures, c=health_colors, s=3, alpha=0.6)
    ax.plot(steps, pressures, linewidth=0.3, alpha=0.3, color="#555")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Pop"),
        Patch(facecolor="#e74c3c", label="Burn"),
        Patch(facecolor="#f1c40f", label="Seep"),
        Patch(facecolor="#bdc3c7", label="Dormant/Insufficient"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pressure (Π)")
    ax.set_title("Diagnostic Timeline")
    ax.grid(True, alpha=0.2)

    if fig:
        fig.tight_layout()
    return fig
