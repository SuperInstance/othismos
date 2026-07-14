# Óthismos — Core Library
#
# The force a bounded system exerts against its bounds.
# The push IS the knowing.
#
# This package implements the mathematical framework described in
# math/01_PRESSURE_MATH.md and the diagnostic framework from
# essays/04_THE_POPCORN_DIAGNOSTIC.md.

from othismos.pressure import (
    PressureMeasurement,
    PressureGauge,
    compute_othismos,
    goldilocks_range,
)
from othismos.phases import (
    MoltPhase,
    PhaseClassifier,
    MoltCycle,
    MoltCycleTracker,
)
from othismos.diagnostics import (
    DiagnosticResult,
    PopcornDiagnostic,
    SystemHealth,
)
from othismos.ecology import (
    Deposit,
    Reef,
    ReefLayer,
    Reefquake,
)

__version__ = "0.1.0"
__all__ = [
    # Pressure
    "PressureMeasurement",
    "PressureGauge",
    "compute_othismos",
    "goldilocks_range",
    # Phases
    "MoltPhase",
    "PhaseClassifier",
    "MoltCycle",
    "MoltCycleTracker",
    # Diagnostics
    "DiagnosticResult",
    "PopcornDiagnostic",
    "SystemHealth",
    # Ecology
    "Deposit",
    "Reef",
    "ReefLayer",
    "Reefquake",
]
