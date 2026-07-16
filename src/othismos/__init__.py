# Óthismos — Core Library
#
# The force a bounded system exerts against its bounds.
# The push IS the knowing.

from othismos.protocols import (
    FeasibilityFn,
    ProjectionFn,
    NormalFn,
    DistanceFn,
    ConstraintLike,
)
from othismos.config import OthismosConfig
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
from othismos.context_pressure import (
    ContextPressureGauge,
    ContextPressureMeasurement,
    cosine_distance,
    l2_distance,
    token_overlap,
)
from othismos.controller import (
    PressureController,
    ControlAction,
    ActionType,
)
from othismos.serialization import (
    save_history,
    load_history,
    save_diagnostic,
    export_metrics_csv,
    pressure_summary,
)

# Optional imports (require extra deps)
try:
    from othismos.viz import (
        plot_pressure,
        plot_molt_cycle,
        plot_constraint_profile,
        plot_diagnostic_timeline,
    )
except ImportError:
    pass

try:
    from othismos.pandas_export import (
        gauge_to_dataframe,
        tracker_to_dataframe,
        reef_to_dataframe,
    )
except ImportError:
    pass

__version__ = "0.4.0"
__all__ = [
    # Protocols
    "FeasibilityFn",
    "ProjectionFn",
    "NormalFn",
    "DistanceFn",
    "ConstraintLike",
    # Config
    "OthismosConfig",
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
    # Context pressure
    "ContextPressureGauge",
    "ContextPressureMeasurement",
    "cosine_distance",
    "l2_distance",
    "token_overlap",
    # Controller
    "PressureController",
    "ControlAction",
    "ActionType",
    # Serialization
    "save_history",
    "load_history",
    "save_diagnostic",
    "export_metrics_csv",
    "pressure_summary",
    # Viz (optional)
    "plot_pressure",
    "plot_molt_cycle",
    "plot_constraint_profile",
    "plot_diagnostic_timeline",
    # Pandas (optional)
    "gauge_to_dataframe",
    "tracker_to_dataframe",
    "reef_to_dataframe",
]
