"""
The Popcorn Diagnostic — classify system health as Pop, Burn, or Seep.

From essays/04_THE_POPCORN_DIAGNOSTIC.md:

- The Pop: pressure builds, hull holds, critical threshold reached, molt.
- The Burn: heat applied but no internal pressure. Silent death.
- The Seep: hull cracked, pressure leaks, no critical event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class SystemHealth(Enum):
    """The three popcorn diagnostic categories plus healthy."""

    POP = "pop"        # Productive pressure, approaching or at phase transition
    BURN = "burn"      # No internal pressure despite external heat. Silent death.
    SEEP = "seep"      # Pressure present but leaking. Activity without progress.
    DORMANT = "dormant"  # No pressure, no heat. Resting. Not dead.

    @property
    def is_healthy(self) -> bool:
        return self in (SystemHealth.POP, SystemHealth.DORMANT)

    @property
    def description(self) -> str:
        return {
            SystemHealth.POP: "Pressure building productively. The system is alive and pushing.",
            SystemHealth.BURN: "No internal pressure. The system looks fine but is hollow. Silent killer.",
            SystemHealth.SEEP: "Pressure leaking through cracks. High activity, no progress.",
            SystemHealth.DORMANT: "No heat applied. The system rests. Waiting for new input.",
        }[self]


@dataclass
class DiagnosticResult:
    """The result of running the popcorn diagnostic."""

    health: SystemHealth
    pressure: float
    heat: float  # external optimization pressure (learning rate, workload, etc.)
    pressure_efficiency: float  # how much push per unit of heat
    leak_rate: float  # estimated rate of pressure loss
    confidence: float
    recommendation: str
    signals: list[str] = field(default_factory=list)


class PopcornDiagnostic:
    """Diagnose system health using the Pop/Burn/Seep framework.

    The diagnostic requires two signals:
    1. **Pressure (Π)** — the óthismos measurement. Internal push.
    2. **Heat** — the external optimization pressure. External force.

    The ratio of pressure to heat determines the diagnosis:
    - High heat, low pressure → Burn (nothing inside to push)
    - High heat, moderate pressure, high variance → Seep (pressure leaking)
    - High heat, high pressure, rising trend → Pop (healthy approach to molt)
    - Low heat, low pressure → Dormant (waiting for input)
    """

    def __init__(
        self,
        burn_threshold: float = 0.01,
        seep_volatility: float = 0.5,
        pop_efficiency: float = 0.3,
    ):
        """
        Args:
            burn_threshold: Pressure below this with heat above threshold = Burn.
            seep_volatility: Coefficient of variation above which = Seep.
            pop_efficiency: Pressure/heat ratio above which = Pop.
        """
        self.burn_threshold = burn_threshold
        self.seep_volatility = seep_volatility
        self.pop_efficiency = pop_efficiency

    def diagnose(
        self,
        pressures: Sequence[float],
        heat: float,
        window: int = 50,
    ) -> DiagnosticResult:
        """Run the diagnostic.

        Args:
            pressures: Chronological óthismos measurements.
            heat: Current external pressure (learning rate × gradient magnitude,
                or workload intensity, or any measure of applied force).
            window: How many recent steps to analyze.

        Returns:
            DiagnosticResult with health classification and recommendation.
        """
        if not pressures or heat < 1e-12:
            return DiagnosticResult(
                health=SystemHealth.DORMANT,
                pressure=0.0,
                heat=heat,
                pressure_efficiency=0.0,
                leak_rate=0.0,
                confidence=0.9,
                recommendation="No external heat applied. System is dormant. Apply input to begin.",
                signals=["heat ≈ 0", "system waiting"],
            )

        recent = list(pressures[-min(len(pressures), window):])
        current = recent[-1]
        mean_p = float(np.mean(recent))
        std_p = float(np.std(recent))
        cv = std_p / mean_p if mean_p > 1e-12 else 0.0  # coefficient of variation

        efficiency = mean_p / heat if heat > 1e-12 else 0.0

        # Trend
        if len(recent) > 1:
            x = np.arange(len(recent))
            trend = float(np.polyfit(x, recent, 1)[0])
        else:
            trend = 0.0

        signals: list[str] = []
        recommendation = ""

        # Classification
        if mean_p < self.burn_threshold and heat > self.burn_threshold:
            health = SystemHealth.BURN
            confidence = 0.85
            signals.append(f"pressure {mean_p:.6f} << heat {heat:.6f}")
            signals.append("internal drive absent — the kernel is dry")
            recommendation = (
                "BURN detected. Stop applying heat. The system has no internal pressure. "
                "Find what the system cares about — a new problem, a new question, "
                "a reason to push. More heat will only burn the kernel further."
            )
        elif cv > self.seep_volatility and mean_p > self.burn_threshold:
            health = SystemHealth.SEEP
            confidence = 0.75
            signals.append(f"pressure volatility (CV={cv:.2f}) > threshold {self.seep_volatility}")
            signals.append("pressure present but unstable — hull is cracked")
            recommendation = (
                "SEEP detected. The system has pressure but it's leaking. "
                "Patch the hull: tighten constraints, reduce scope, close the system. "
                "Stop adding energy — it leaks out faster than it accumulates. "
                "One project, one constraint, no new ideas until pressure stabilizes."
            )
        elif efficiency >= self.pop_efficiency and trend >= 0:
            health = SystemHealth.POP
            confidence = 0.80
            signals.append(f"pressure efficiency {efficiency:.2f} >= {self.pop_efficiency}")
            if trend > 0:
                signals.append(f"pressure rising (slope={trend:.6f}) — approaching critical threshold")
            else:
                signals.append("pressure stable — maintaining productive contact with walls")
            if trend > 0:
                recommendation = (
                    "POP trajectory. Pressure building productively. Maintain temperature. "
                    "The kernel knows what it's doing. The push is coming."
                )
            else:
                recommendation = (
                    "Healthy POP. Productive pressure in the Goldilocks zone. "
                    "The system is doing its best work. Continue."
                )
        else:
            # Borderline case
            if efficiency < self.pop_efficiency * 0.5:
                health = SystemHealth.BURN
                confidence = 0.55
                signals.append(f"low efficiency ({efficiency:.4f}) — borderline Burn")
                recommendation = (
                    "Borderline BURN. Pressure is very low relative to heat. "
                    "Consider reducing heat or finding a more engaging problem."
                )
            else:
                health = SystemHealth.SEEP
                confidence = 0.55
                signals.append(f"moderate efficiency but unstable ({efficiency:.4f}) — borderline Seep")
                recommendation = (
                    "Borderline SEEP. Pressure is present but not accumulating efficiently. "
                    "Check for constraint leaks — tasks that dissipate energy without progress."
                )

        return DiagnosticResult(
            health=health,
            pressure=current,
            heat=heat,
            pressure_efficiency=efficiency,
            leak_rate=cv if health == SystemHealth.SEEP else 0.0,
            confidence=confidence,
            recommendation=recommendation,
            signals=signals,
        )
