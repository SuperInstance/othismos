"""Tests for the molt cycle phase classifier."""

import pytest

from othismos.phases import (
    MoltPhase,
    PhaseClassifier,
    PhaseReading,
    MoltCycle,
    MoltCycleTracker,
)


class TestPhaseClassifier:
    def test_empty_history(self):
        clf = PhaseClassifier()
        reading = clf.classify([])
        assert reading.phase == MoltPhase.DORMANCY
        assert reading.pressure == 0.0

    def test_dormancy(self):
        clf = PhaseClassifier()
        reading = clf.classify([0.0, 0.0, 0.0, 0.0, 0.0])
        assert reading.phase == MoltPhase.DORMANCY

    def test_resistance(self):
        clf = PhaseClassifier(crisis_threshold=1.0, expansion_floor=0.1)
        pressures = [0.3, 0.35, 0.4, 0.38, 0.42, 0.4, 0.41, 0.39]
        reading = clf.classify(pressures)
        assert reading.phase == MoltPhase.RESISTANCE
        assert reading.confidence > 0.5

    def test_expansion(self):
        clf = PhaseClassifier(crisis_threshold=1.0, expansion_floor=0.5)
        # Low pressure but rising
        pressures = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
        reading = clf.classify(pressures)
        assert reading.phase == MoltPhase.EXPANSION

    def test_crisis(self):
        clf = PhaseClassifier(crisis_threshold=0.5, expansion_floor=0.1)
        # High and rising pressure
        pressures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        reading = clf.classify(pressures)
        assert reading.phase == MoltPhase.CRISIS

    def test_phase_descriptions(self):
        for phase in MoltPhase:
            assert isinstance(phase.description, str)
            assert len(phase.description) > 0

    def test_phase_labels(self):
        assert MoltPhase.RESISTANCE.label == "Resistance"
        assert MoltPhase.CRISIS.label == "Crisis"


class TestMoltCycleTracker:
    def test_initial_state(self):
        tracker = MoltCycleTracker()
        assert tracker.current_phase is None
        assert tracker.cycle_count == 0

    def test_single_phase_tracking(self):
        tracker = MoltCycleTracker(
            classifier=PhaseClassifier(crisis_threshold=1.0, expansion_floor=0.1)
        )
        tracker.update(0.4)
        assert tracker.current_phase is not None

    def test_cycle_detection(self):
        classifier = PhaseClassifier(crisis_threshold=0.8, expansion_floor=0.2)
        tracker = MoltCycleTracker(classifier=classifier)

        # Simulate a full cycle: low → high → drop → low
        pressures = (
            [0.1, 0.15, 0.2, 0.3]      # Expansion
            + [0.4, 0.5, 0.6, 0.7]      # Resistance
            + [0.8, 0.9, 1.0, 1.1]      # Crisis
            + [0.3, 0.15, 0.1]          # Settlement/Expansion
        )

        for p in pressures:
            tracker.update(p)

        assert tracker.cycle_count >= 1

    def test_staircase_metric(self):
        tracker = MoltCycleTracker()

        # Not enough cycles
        result = tracker.staircase_metric()
        assert result["health"] == "insufficient data"

    def test_multiple_cycles_staircase(self):
        classifier = PhaseClassifier(crisis_threshold=0.8, expansion_floor=0.2)
        tracker = MoltCycleTracker(classifier=classifier)

        # Two cycles
        for cycle in range(2):
            for p in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.3, 0.15, 0.1]:
                tracker.update(p)

        result = tracker.staircase_metric()
        assert "cycles" in result
        assert "health" in result
        assert result["cycles"] >= 1


class TestMoltCycleDataclass:
    def test_empty_cycle(self):
        cycle = MoltCycle(cycle_number=0, start_step=0)
        assert cycle.duration == 0
        assert cycle.peak_pressure == 0.0
        assert cycle.phase_sequence == []
