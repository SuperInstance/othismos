#!/usr/bin/env python3
"""
Óthismos Demo — Watch a bounded system push against its walls.

This demo simulates a simple optimization process with L2 constraints
and shows óthismos (pressure), molt phase, and popcorn diagnostic
at each stage of the cycle.
"""

import sys
import numpy as np

sys.path.insert(0, "src")

from othismos import (
    PressureGauge,
    PhaseClassifier,
    MoltCycleTracker,
    PopcornDiagnostic,
)
from othismos.pressure import l2_constraint


def simulate_cycle():
    """Simulate a bounded optimizer going through a full molt cycle."""

    # System state
    theta = np.array([0.0, 0.0])
    true_minimum = np.array([5.0, 5.0])  # Outside the constraint set!

    # Constraint: L2 ball of radius 1.0
    constraint = l2_constraint("weight_budget", radius=1.0)

    # Instruments
    gauge = PressureGauge(window_size=2000)
    classifier = PhaseClassifier(crisis_threshold=0.08, expansion_floor=0.02)
    tracker = MoltCycleTracker(classifier=classifier)
    diagnostic = PopcornDiagnostic(burn_threshold=0.001, pop_efficiency=0.1)

    learning_rate = 0.05
    all_pressures = []

    print("=" * 70)
    print("ÓTHISMOS DEMO — A Bounded System Pushing Against Its Walls")
    print("=" * 70)
    print(f"\nConstraint: L2 ball, radius={1.0}")
    print(f"True minimum: {true_minimum} (outside the ball!)")
    print(f"Learning rate: {learning_rate}")
    print(f"Starting point: {theta}")
    print()

    for step in range(500):
        # Gradient of squared distance to true minimum
        gradient = 2.0 * (theta - true_minimum)

        # Measure pressure
        m = gauge.measure(theta, gradient, learning_rate, [constraint])
        all_pressures.append(m.pressure)

        # Take the actual constrained step
        theta = theta + m.actual_step

        # Phase tracking
        reading = tracker.update(m.pressure)

        # Report at key moments
        if step in (0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 499):
            diag = diagnostic.diagnose(all_pressures, heat=learning_rate * np.linalg.norm(gradient))
            print(f"Step {step:4d} | Π={m.pressure:.6f} | "
                  f"θ=({theta[0]:+.4f}, {theta[1]:+.4f}) | "
                  f"Phase: {reading.phase.label:12s} (conf={reading.confidence:.2f}) | "
                  f"Health: {diag.health.value.upper()}")
            if reading.signals:
                for sig in reading.signals[:2]:
                    print(f"           → {sig}")
            print()

    # Final reports
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(f"\nTotal steps: 500")
    print(f"Final θ: ({theta[0]:+.6f}, {theta[1]:+.6f})")
    print(f"Distance to true min: {np.linalg.norm(theta - true_minimum):.4f}")
    print(f"Distance to constraint boundary: {1.0 - np.linalg.norm(theta):.4f}")

    print(f"\n--- Pressure Stats ---")
    print(f"Mean Π:   {gauge.mean_pressure:.6f}")
    print(f"Peak Π:   {max(all_pressures):.6f}")
    print(f"Final Π:  {gauge.current_pressure:.6f}")
    print(f"Trend:    {gauge.pressure_trend:+.6f}")
    zone = gauge.goldilocks()
    print(f"Goldilocks: [{zone.lower_bound:.6f}, {zone.upper_bound:.6f}]")

    profile = gauge.pressure_profile()
    if profile:
        print(f"By constraint:")
        for name, val in profile.items():
            print(f"  {name}: {val:.6f}")

    print(f"\n--- Molt Cycle ---")
    print(f"Cycles detected: {tracker.cycle_count}")
    print(f"Current phase:   {tracker.current_phase.label if tracker.current_phase else 'None'}")
    staircase = tracker.staircase_metric()
    for k, v in staircase.items():
        print(f"  {k}: {v}")

    print(f"\n--- Diagnostic ---")
    final_diag = diagnostic.diagnose(all_pressures, heat=0.01)
    print(f"Health: {final_diag.health.value.upper()}")
    print(f"Pressure efficiency: {final_diag.pressure_efficiency:.4f}")
    print(f"Confidence: {final_diag.confidence:.2f}")
    print(f"Recommendation: {final_diag.recommendation}")

    print()
    print("The push IS the knowing.")


if __name__ == "__main__":
    simulate_cycle()
