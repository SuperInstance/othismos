"""
Configuration objects for óthismos.

Single dataclass to configure gauge + tracker + diagnostic + controller
together. Supports YAML/dict construction for config-file-driven workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(slots=True)
class OthismosConfig:
    """Configuration for the full óthismos monitoring stack.

    Pass to PressureGauge, MoltCycleTracker, PopcornDiagnostic, and
    PressureController to ensure consistent settings.

    Example:
        >>> config = OthismosConfig(crisis_threshold=0.5, window_size=5000)
        >>> gauge = PressureGauge(window_size=config.window_size)
        >>> classifier = PhaseClassifier(
        ...     crisis_threshold=config.crisis_threshold,
        ...     expansion_floor=config.expansion_floor,
        ... )

    Or from dict/YAML:
        >>> config = OthismosConfig.from_dict({"crisis_threshold": 0.5})
        >>> config = OthismosConfig.from_yaml("config.yaml")
    """

    # Pressure gauge
    window_size: int = 1000

    # Phase classifier
    crisis_threshold: float | None = None
    expansion_floor: float | None = None
    settlement_decline_rate: float = -0.5

    # Popcorn diagnostic
    burn_threshold: float = 0.01
    seep_volatility: float = 0.5
    pop_efficiency: float = 0.3

    # Controller
    lr_bounds: tuple[float, float] = (1e-6, 1.0)
    crisis_lr_factor: float = 0.5
    expansion_lr_factor: float = 1.3
    burn_patience: int = 50

    # Logging
    log_every: int = 1
    log_backend: str = "dict"  # dict | wandb | tensorboard | stdout

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OthismosConfig":
        """Create config from a dictionary."""
        valid = {k: v for k, v in data.items() if k in cls.__annotations__ or k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_yaml(cls, path: str) -> "OthismosConfig":
        """Create config from a YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write config to a YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def build_gauge(self):
        """Construct a PressureGauge from this config."""
        from othismos.pressure import PressureGauge
        return PressureGauge(window_size=self.window_size)

    def build_classifier(self):
        """Construct a PhaseClassifier from this config."""
        from othismos.phases import PhaseClassifier
        return PhaseClassifier(
            crisis_threshold=self.crisis_threshold,
            expansion_floor=self.expansion_floor,
            settlement_decline_rate=self.settlement_decline_rate,
        )

    def build_diagnostic(self):
        """Construct a PopcornDiagnostic from this config."""
        from othismos.diagnostics import PopcornDiagnostic
        return PopcornDiagnostic(
            burn_threshold=self.burn_threshold,
            seep_volatility=self.seep_volatility,
            pop_efficiency=self.pop_efficiency,
        )

    def build_controller(self, gauge=None, tracker=None):
        """Construct a PressureController from this config."""
        from othismos.controller import PressureController
        from othismos.phases import MoltCycleTracker
        gauge = gauge or self.build_gauge()
        tracker = tracker or MoltCycleTracker(classifier=self.build_classifier())
        return PressureController(
            gauge=gauge,
            tracker=tracker,
            lr_bounds=self.lr_bounds,
            crisis_lr_factor=self.crisis_lr_factor,
            expansion_lr_factor=self.expansion_lr_factor,
            burn_patience=self.burn_patience,
        )

    def build_all(self):
        """Build the full stack: gauge, tracker, diagnostic, controller.

        Returns:
            (gauge, tracker, diagnostic, controller)
        """
        gauge = self.build_gauge()
        classifier = self.build_classifier()
        from othismos.phases import MoltCycleTracker
        tracker = MoltCycleTracker(classifier=classifier)
        diagnostic = self.build_diagnostic()
        controller = self.build_controller(gauge=gauge, tracker=tracker)
        return gauge, tracker, diagnostic, controller
