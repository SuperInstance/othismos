"""Tests for the popcorn diagnostic."""

import pytest

from othismos.diagnostics import (
    SystemHealth,
    DiagnosticResult,
    PopcornDiagnostic,
)


class TestPopcornDiagnostic:
    def test_dormant_no_heat(self):
        diag = PopcornDiagnostic()
        result = diag.diagnose(pressures=[], heat=0.0)
        assert result.health == SystemHealth.DORMANT
        assert result.pressure == 0.0

    def test_dormant_no_pressure_history(self):
        diag = PopcornDiagnostic()
        result = diag.diagnose(pressures=[], heat=1.0)
        assert result.health == SystemHealth.DORMANT

    def test_burn(self):
        """High heat, very low pressure → Burn."""
        diag = PopcornDiagnostic(burn_threshold=0.1, pop_efficiency=0.3)
        pressures = [0.001, 0.002, 0.001, 0.001, 0.002]
        result = diag.diagnose(pressures, heat=1.0)
        assert result.health == SystemHealth.BURN
        assert "BURN" in result.recommendation

    def test_seep(self):
        """Moderate pressure but highly volatile → Seep."""
        diag = PopcornDiagnostic(burn_threshold=0.01, seep_volatility=0.3, pop_efficiency=0.5)
        pressures = [0.1, 0.5, 0.05, 0.8, 0.02, 0.6, 0.01, 0.7]
        result = diag.diagnose(pressures, heat=1.0)
        assert result.health == SystemHealth.SEEP
        assert "SEEP" in result.recommendation

    def test_pop_stable(self):
        """Good pressure efficiency, stable → Pop."""
        diag = PopcornDiagnostic(burn_threshold=0.01, seep_volatility=0.5, pop_efficiency=0.3)
        pressures = [0.4, 0.42, 0.41, 0.43, 0.42, 0.44, 0.43, 0.42]
        result = diag.diagnose(pressures, heat=1.0)
        assert result.health == SystemHealth.POP

    def test_pop_rising(self):
        """Rising pressure with good efficiency → Pop."""
        diag = PopcornDiagnostic(burn_threshold=0.01, seep_volatility=0.5, pop_efficiency=0.2)
        pressures = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        result = diag.diagnose(pressures, heat=0.5)
        assert result.health == SystemHealth.POP
        assert "rising" in result.recommendation.lower() or "productive" in result.recommendation.lower()

    def test_health_properties(self):
        assert SystemHealth.POP.is_healthy
        assert SystemHealth.DORMANT.is_healthy
        assert not SystemHealth.BURN.is_healthy
        assert not SystemHealth.SEEP.is_healthy

    def test_descriptions(self):
        for health in SystemHealth:
            assert isinstance(health.description, str)
            assert len(health.description) > 10

    def test_pressure_efficiency(self):
        diag = PopcornDiagnostic()
        pressures = [0.5, 0.5, 0.5, 0.5]
        result = diag.diagnose(pressures, heat=1.0)
        assert result.pressure_efficiency > 0

    def test_confidence_in_range(self):
        diag = PopcornDiagnostic()
        for pressures, heat in [
            ([0.001, 0.002], 1.0),
            ([0.5, 0.5, 0.5], 1.0),
            ([0.1, 0.5, 0.05, 0.8], 1.0),
        ]:
            result = diag.diagnose(pressures, heat)
            assert 0.0 <= result.confidence <= 1.0
