"""Tests for óthismos framework integrations."""

import numpy as np
import pytest

from othismos.pressure import l2_constraint, box_constraint
from othismos.integrations import (
    DictLogger,
    OthismosTorchCallback,
    constraint_from_torch_model,
)
from othismos.serialization import pressure_summary


class TestDictLogger:
    def test_log_and_history(self):
        logger = DictLogger()
        logger.log({"loss": 0.5, "pressure": 0.1}, step=0)
        logger.log({"loss": 0.3, "pressure": 0.2}, step=1)

        assert len(logger.history) == 2
        assert logger.history[0][1]["loss"] == 0.5

    def test_metric_series(self):
        logger = DictLogger()
        logger.log({"pressure": 0.1}, step=0)
        logger.log({"pressure": 0.3}, step=1)
        logger.log({"pressure": 0.5}, step=2)

        series = logger.metric_series("pressure")
        assert series == [0.1, 0.3, 0.5]


class TestOthismosCallback:
    def test_construction(self):
        constraints = [l2_constraint("l2", radius=1.0)]
        cb = OthismosTorchCallback(constraints=constraints)
        assert len(cb.constraints) == 1
        assert cb.gauge is not None
        assert cb.tracker is not None

    def test_construction_with_logger(self):
        logger = DictLogger()
        constraints = [l2_constraint("l2", radius=1.0)]
        cb = OthismosTorchCallback(constraints=constraints, logger=logger, log_every=5)
        assert cb.logger is logger
        assert cb.log_every == 5

    def test_health_report_empty(self):
        constraints = [l2_constraint("l2", radius=1.0)]
        cb = OthismosTorchCallback(constraints=constraints)
        report = cb.health_report()
        assert "No measurements" in report

    def test_health_report_with_data(self):
        constraints = [l2_constraint("l2", radius=0.1)]
        cb = OthismosTorchCallback(constraints=constraints)

        # Simulate some pressure readings
        for i in range(20):
            from othismos.pressure import PressureMeasurement
            m = PressureMeasurement(
                step=i,
                desired_step=np.array([0.1, 0.0]),
                actual_step=np.array([0.05, 0.0]),
                violation=np.array([0.05, 0.0]),
                pressure=0.05 + i * 0.001,
            )
            cb.gauge._history.append(m)
            cb.gauge._step += 1
            cb.tracker.update(m.pressure)

        report = cb.health_report()
        assert "Health Report" in report
        assert "POP" in report or "SEEP" in report or "BURN" in report or "DORMANT" in report


class TestConstraintBuilder:
    def test_max_norm_constraint(self):
        class FakeModel:
            def parameters(self):
                class P:
                    def __init__(self, n):
                        self.n = n
                    def numel(self):
                        return self.n
                return [P(10), P(20)]

        constraints = constraint_from_torch_model(FakeModel(), max_norm=1.0)
        assert len(constraints) == 1
        assert constraints[0].name == "global_l2"

    def test_max_per_param_constraint(self):
        class FakeModel:
            def parameters(self):
                class P:
                    def __init__(self, n):
                        self.n = n
                    def numel(self):
                        return self.n
                return [P(10)]

        constraints = constraint_from_torch_model(FakeModel(), max_per_param=0.5)
        assert len(constraints) == 1
        assert constraints[0].name == "per_param_clip"

    def test_no_constraints(self):
        class FakeModel:
            def parameters(self):
                return []
        constraints = constraint_from_torch_model(FakeModel())
        assert len(constraints) == 0
