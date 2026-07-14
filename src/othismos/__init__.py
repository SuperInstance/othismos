# Óthismos — Core Library
#
# The force a bounded system exerts against its bounds.
# The push IS the knowing.
#
# This package implements the mathematical framework described in
# math/01_PRESSURE_MATH.md and the diagnostic framework from
# essays/04_THE_POPCORN_DIAGNOSTIC.md.

from othismos.pressure import (
    ConstraintType,
    Constraint,
    PressureMeasurement,
    PressureGauge,
    GoldilocksZone,
    compute_othismos,
    goldilocks_range,
    l2_constraint,
    box_constraint,
)
from othismos.phases import (
    MoltPhase,
    PhaseClassifier,
    PhaseReading,
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
from othismos.integrations import (
    MetricLogger,
    DictLogger,
    OthismosTorchCallback,
    OthismosTrainerCallback,
    constraint_from_torch_model,
)
from othismos.serialization import (
    save_history,
    load_history,
    save_diagnostic,
    export_metrics_csv,
    pressure_summary,
)

__version__ = "0.2.0"
__all__ = [
    # Pressure
    "ConstraintType",
    "Constraint",
    "PressureMeasurement",
    "PressureGauge",
    "GoldilocksZone",
    "compute_othismos",
    "goldilocks_range",
    "l2_constraint",
    "box_constraint",
    # Phases
    "MoltPhase",
    "PhaseClassifier",
    "PhaseReading",
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
    # Integrations
    "MetricLogger",
    "DictLogger",
    "OthismosTorchCallback",
    "OthismosTrainerCallback",
    "constraint_from_torch_model",
    # Serialization
    "save_history",
    "load_history",
    "save_diagnostic",
    "export_metrics_csv",
    "pressure_summary",
]
