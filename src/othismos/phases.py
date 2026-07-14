"""
Molt cycle phase detection and tracking.

Implements the five-phase molt cycle from ecology/02_THE_MOLT_CYCLE.md:
Expansion → Resistance → Crisis → Settlement → (Dormancy)

Each phase is classified by óthismos (Π) level and trend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np


class MoltPhase(IntEnum):
    """The five phases of the molt cycle."""

    DORMANCY = 0      # Π ≈ 0, waiting for new heat
    EXPANSION = 1     # Π low but rising, lots of headroom
    RESISTANCE = 2    # Π high and stable, walls reached, quality work
    CRISIS = 3        # Π at peak, structural failure imminent, molt approaching
    SETTLEMENT = 4    # Π dropped sharply post-molt, new envelope stabilizing

    @property
    def label(self) -> str:
        return self.name.title()

    @property
    def description(self) -> str:
        descriptions = {
            MoltPhase.DORMANCY: "Suspended óthismos. Alive but not pushing. Waiting for new heat.",
            MoltPhase.EXPANSION: "Fresh constraint envelope. Rapid growth, low pressure. Filling space.",
            MoltPhase.RESISTANCE: "Walls reached. Best work happens here. Pressure forces precision.",
            MoltPhase.CRISIS: "Pressure exceeds structural tolerance. Molt or rupture. Most dangerous phase.",
            MoltPhase.SETTLEMENT: "Post-molt. New envelope soft and vulnerable. Solidifying.",
        }
        return descriptions.get(self, "")


@dataclass
class PhaseReading:
    """A single phase classification reading."""

    step: int
    phase: MoltPhase
    pressure: float
    confidence: float  # 0.0 to 1.0
    signals: list[str] = field(default_factory=list)


class PhaseClassifier:
    """Classify the current molt phase from pressure dynamics.

    Uses pressure level, trend, and volatility to determine phase.
    """

    def __init__(
        self,
        crisis_threshold: float | None = None,
        expansion_floor: float | None = None,
        settlement_decline_rate: float = -0.5,
    ):
        """
        Args:
            crisis_threshold: Π above which the system is in Crisis.
                If None, auto-calibrated from history.
            expansion_floor: Π below which (with rising trend) the system
                is in Expansion. If None, auto-calibrated.
            settlement_decline_rate: Pressure slope below which (after
                high pressure) indicates Settlement.
        """
        self.crisis_threshold = crisis_threshold
        self.expansion_floor = expansion_floor
        self.settlement_decline_rate = settlement_decline_rate
        self._last_high_pressure_step: int | None = None

    def classify(
        self,
        pressures: Sequence[float],
        window: int = 50,
    ) -> PhaseReading:
        """Classify the current phase from a pressure history.

        Args:
            pressures: Chronological list of Π values.
            window: Number of recent steps to analyze.

        Returns:
            PhaseReading with phase, confidence, and diagnostic signals.
        """
        n = len(pressures)
        if n == 0:
            return PhaseReading(0, MoltPhase.DORMANCY, 0.0, 0.0, ["no data"])

        recent = list(pressures[-min(n, window):])
        current = recent[-1]
        signals: list[str] = []

        # Compute statistics
        mean_p = float(np.mean(recent))
        std_p = float(np.std(recent))
        x = np.arange(len(recent))
        trend = float(np.polyfit(x, recent, 1)[0]) if len(recent) > 1 else 0.0

        # Auto-calibrate thresholds if not set
        crisis_th = self.crisis_threshold if self.crisis_threshold is not None else (mean_p + 2 * std_p if std_p > 0 else mean_p * 1.5)
        expansion_th = self.expansion_floor if self.expansion_floor is not None else (mean_p * 0.3)

        # Classification logic
        confidence = 0.5
        phase = MoltPhase.RESISTANCE  # default

        if current < 1e-8:
            phase = MoltPhase.DORMANCY
            confidence = 0.95
            signals.append("pressure near zero")
        elif current >= crisis_th and trend >= 0:
            phase = MoltPhase.CRISIS
            confidence = 0.85
            signals.append(f"pressure {current:.4f} >= crisis threshold {crisis_th:.4f}")
            signals.append("pressure still rising or flat — molt approaching")
            self._last_high_pressure_step = n
        elif self._last_high_pressure_step is not None:
            steps_since_crisis = n - self._last_high_pressure_step
            if steps_since_crisis < window // 2 and trend < self.settlement_decline_rate:
                phase = MoltPhase.SETTLEMENT
                confidence = 0.80
                signals.append(f"pressure declining after crisis ({steps_since_crisis} steps ago)")
                signals.append(f"slope {trend:.6f} < settlement rate {self.settlement_decline_rate}")
            elif steps_since_crisis < window:
                phase = MoltPhase.EXPANSION
                confidence = 0.70
                signals.append(f"post-crisis ({steps_since_crisis} steps), pressure low and stabilizing")
            else:
                self._last_high_pressure_step = None
                phase = MoltPhase.RESISTANCE
                confidence = 0.65
                signals.append("pressure stabilized post-crisis, walls reached again")
        elif current <= expansion_th and trend > 0:
            phase = MoltPhase.EXPANSION
            confidence = 0.75
            signals.append(f"pressure {current:.4f} <= expansion floor {expansion_th:.4f}")
            signals.append("trend rising — system is filling available space")
        elif current > expansion_th and current < crisis_th:
            phase = MoltPhase.RESISTANCE
            confidence = 0.80
            signals.append(f"pressure in Goldilocks zone [{expansion_th:.4f}, {crisis_th:.4f}]")
            if std_p > 0:
                signals.append(f"pressure stable (σ={std_p:.4f}) — productive constraint contact")
        else:
            # Pressure high but declining — could be late Crisis or settling
            if trend < 0:
                phase = MoltPhase.SETTLEMENT
                confidence = 0.60
                signals.append("high pressure but declining — possible settlement")
            else:
                phase = MoltPhase.CRISIS
                confidence = 0.55
                signals.append("high pressure, trend unclear — assume crisis")

        return PhaseReading(
            step=n - 1,
            phase=phase,
            pressure=current,
            confidence=confidence,
            signals=signals,
        )


@dataclass
class MoltCycle:
    """A complete molt cycle record.

    Tracks one trip through all five phases.
    """

    cycle_number: int
    start_step: int
    phases: list[PhaseReading] = field(default_factory=list)

    @property
    def end_step(self) -> int:
        return self.phases[-1].step if self.phases else self.start_step

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step

    @property
    def peak_pressure(self) -> float:
        return max((r.pressure for r in self.phases), default=0.0)

    @property
    def phase_sequence(self) -> list[MoltPhase]:
        return [r.phase for r in self.phases]


class MoltCycleTracker:
    """Track molt cycles over time.

    Detects cycle boundaries (Crisis → Settlement transitions)
    and maintains historical records for the staircase metric.
    """

    def __init__(self, classifier: PhaseClassifier | None = None):
        self.classifier = classifier or PhaseClassifier()
        self.cycles: list[MoltCycle] = []
        self._current_cycle: MoltCycle | None = None
        self._all_pressures: list[float] = []
        self._cycle_count = 0
        self._last_phase: MoltPhase | None = None

    def update(self, pressure: float) -> PhaseReading:
        """Record a pressure reading and update cycle tracking."""
        self._all_pressures.append(pressure)
        reading = self.classifier.classify(self._all_pressures)

        # Detect cycle boundary: Crisis → Settlement
        if (
            self._last_phase == MoltPhase.CRISIS
            and reading.phase in (MoltPhase.SETTLEMENT, MoltPhase.EXPANSION)
        ):
            # Close current cycle, start new one
            if self._current_cycle:
                self.cycles.append(self._current_cycle)
            self._cycle_count += 1
            self._current_cycle = MoltCycle(
                cycle_number=self._cycle_count,
                start_step=reading.step,
            )
        elif self._current_cycle is None:
            self._current_cycle = MoltCycle(
                cycle_number=self._cycle_count,
                start_step=reading.step,
            )

        self._current_cycle.phases.append(reading)
        self._last_phase = reading.phase
        return reading

    @property
    def current_phase(self) -> MoltPhase | None:
        if self._current_cycle and self._current_cycle.phases:
            return self._current_cycle.phases[-1].phase
        return None

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def staircase_metric(self) -> dict:
        """Compute the staircase metric across all completed cycles.

        Returns:
            Dict with periodicity, peak pressures, durations,
            and health assessment.
        """
        if len(self.cycles) < 2:
            return {
                "cycles": len(self.cycles),
                "periodicity": None,
                "health": "insufficient data",
            }

        durations = [c.duration for c in self.cycles]
        peaks = [c.peak_pressure for c in self.cycles]

        duration_std = float(np.std(durations)) if len(durations) > 1 else 0.0
        peak_std = float(np.std(peaks)) if len(peaks) > 1 else 0.0
        mean_duration = float(np.mean(durations))
        periodicity = 1.0 - (duration_std / mean_duration) if mean_duration > 0 else 0.0

        # Health assessment
        if periodicity > 0.7 and peak_std / max(np.mean(peaks), 1e-12) < 0.3:
            health = "healthy — regular periodicity, consistent crisis thresholds"
        elif periodicity < 0.3:
            health = "irregular — compressed or stalled cycles, investigate"
        elif peak_std / max(np.mean(peaks), 1e-12) > 0.5:
            health = "unstable — crisis thresholds vary significantly"
        else:
            health = "moderate — some variation in cycle structure"

        return {
            "cycles": len(self.cycles),
            "mean_duration": mean_duration,
            "duration_std": duration_std,
            "periodicity": periodicity,
            "mean_peak_pressure": float(np.mean(peaks)),
            "peak_pressure_std": peak_std,
            "health": health,
        }
