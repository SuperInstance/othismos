"""Tests for the pressure-driven adaptive controller."""

import numpy as np
import pytest

from othismos.pressure import (
    PressureGauge,
    PressureMeasurement,
    l2_constraint,
)
from othismos.phases import MoltCycleTracker, PhaseClassifier, MoltPhase
from othismos.diagnostics import SystemHealth
from othismos.controller import (
    PressureController,
    ControlAction,
    ActionType,
)


def inject_pressure(gauge: PressureGauge, pressure: float, step: int = -1):
    """Inject a fake pressure measurement into a gauge."""
    m = PressureMeasurement(
        step=step if step >= 0 else len(gauge.history),
        desired_step=np.array([pressure, 0.0]),
        actual_step=np.array([0.0, 0.0]),
        violation=np.array([pressure, 0.0]),
        pressure=pressure,
    )
    gauge._history.append(m)
    gauge._step += 1


class TestPressureController:
    def test_construction(self):
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge)
        assert ctrl.gauge is gauge
        assert ctrl.lr_bounds == (1e-6, 1.0)

    def test_empty_gauge_no_actions(self):
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge)
        actions = ctrl.update(current_lr=0.01)
        assert actions == []

    def test_crisis_reduces_lr(self):
        classifier = PhaseClassifier(crisis_threshold=0.3, expansion_floor=0.05)
        tracker = MoltCycleTracker(classifier=classifier)
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge, tracker=tracker, crisis_lr_factor=0.5)

        # Simulate rising pressure into crisis
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            inject_pressure(gauge, p)
            ctrl.tracker.update(p)

        actions = ctrl.update(current_lr=0.01)
        lr_actions = [a for a in actions if a.type == ActionType.ADJUST_LR]
        assert len(lr_actions) > 0
        assert lr_actions[0].factor == 0.5

    def test_expansion_increases_lr(self):
        classifier = PhaseClassifier(crisis_threshold=1.0, expansion_floor=0.5)
        tracker = MoltCycleTracker(classifier=classifier)
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge, tracker=tracker, expansion_lr_factor=1.5)

        # Start with no pressure (dormant), then add low rising pressure
        for p in [0.01, 0.02, 0.03, 0.04, 0.05]:
            inject_pressure(gauge, p)
            ctrl.tracker.update(p)

        actions = ctrl.update(current_lr=0.01)
        lr_actions = [a for a in actions if a.type == ActionType.ADJUST_LR]
        # Should suggest increasing LR in expansion
        assert len(lr_actions) > 0
        assert lr_actions[0].factor > 1.0

    def test_burn_alerts_after_patience(self):
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge, burn_patience=3)

        # Simulate burn: high heat but near-zero pressure
        # Call update per-step so burn counter accumulates
        actions_all = []
        for _ in range(5):
            inject_pressure(gauge, 0.0001)
            actions = ctrl.update(current_lr=0.01, heat=1.0)
            actions_all.extend(actions)

        alerts = [a for a in actions_all if a.type == ActionType.ALERT]
        assert len(alerts) > 0
        assert "BURN" in alerts[0].message

    def test_lr_clamped_to_bounds(self):
        gauge = PressureGauge()
        ctrl = PressureController(
            gauge=gauge,
            lr_bounds=(0.001, 0.1),
            crisis_lr_factor=0.5,
        )
        classifier = PhaseClassifier(crisis_threshold=0.3, expansion_floor=0.05)
        ctrl.tracker = MoltCycleTracker(classifier=classifier)

        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            inject_pressure(gauge, p)
            ctrl.tracker.update(p)

        # Even with crisis LR reduction, can't go below 0.001
        actions = ctrl.update(current_lr=0.001)
        lr_actions = [a for a in actions if a.type == ActionType.ADJUST_LR]
        # LR already at minimum, shouldn't suggest reducing further
        if lr_actions:
            assert lr_actions[0].factor * 0.001 >= 0.001 - 1e-12

    def test_checkpoint_before_crisis(self):
        classifier = PhaseClassifier(crisis_threshold=0.3, expansion_floor=0.05)
        tracker = MoltCycleTracker(classifier=classifier)
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge, tracker=tracker)

        # Go from resistance to crisis
        inject_pressure(gauge, 0.25)
        ctrl.tracker.update(0.25)

        inject_pressure(gauge, 0.5)
        ctrl.tracker.update(0.5)

        actions = ctrl.update(current_lr=0.01)
        checkpoints = [a for a in actions if a.type == ActionType.CHECKPOINT]
        assert len(checkpoints) > 0

    def test_status(self):
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge)

        for i in range(20):
            inject_pressure(gauge, 0.1 + i * 0.01)
            ctrl.tracker.update(0.1 + i * 0.01)

        status = ctrl.status()
        assert "step" in status
        assert "phase" in status
        assert "current_pressure" in status
        assert "burn_counter" in status

    def test_actions_sorted_by_priority(self):
        gauge = PressureGauge()
        ctrl = PressureController(gauge=gauge, burn_patience=1)

        # Create conditions for multiple actions
        for _ in range(5):
            inject_pressure(gauge, 0.0001)  # burn condition

        actions = ctrl.update(current_lr=0.01, heat=1.0)
        if len(actions) > 1:
            for i in range(len(actions) - 1):
                assert actions[i].priority >= actions[i + 1].priority
