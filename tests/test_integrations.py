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

    def test_apply_constraints_flag(self):
        """Test that apply_constraints flag writes back constrained parameters."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        # Simple linear model: y = w * x + b
        model = nn.Linear(2, 1, bias=True)
        # Set initial weights and bias to known values
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)

        # Constraint: L2 ball radius 0.5 around origin (constraint on all parameters)
        # Flattened parameters: [w1, w2, b]
        constraint = l2_constraint("weight_budget", radius=0.5)
        constraints = [constraint]

        # Callback with apply_constraints=True
        callback = OthismosTorchCallback(constraints=constraints, apply_constraints=True)

        # Simulate a pre-step (capture current parameters)
        callback.pre_step(model, None, None)  # optimizer and loss not used in our implementation
        pre_params = callback._pre_step_params.copy()
        # Flattened params should be [1.0, 1.0, 0.0]
        expected_pre = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(pre_params, expected_pre)

        # Simulate an optimizer step that would take parameters outside the constraint
        # We'll manually set the post-step parameters to a value outside the radius
        # For example, [2.0, 2.0, 0.0] has norm sqrt(8) ~ 2.828 > 0.5
        # To simulate this, we set the model parameters directly to that value
        # (as if optimizer.step() had already been called)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 2.0]]))
            model.bias.copy_(torch.tensor([0.0]))

        # Now call post_step, which will measure pressure and optionally apply constraints
        metrics = callback.post_step(model)

        # If apply_constraints=True, the model parameters should be projected back to the constraint set
        # The projected value should be the closest point on the L2 ball of radius 0.5 to [2,2,0]
        # Direction is [2,2,0]; unit vector is [1/sqrt(2), 1/sqrt(2), 0] ~ [0.7071,0.7071,0]
        # Multiply by radius 0.5 gives [0.3536, 0.3536, 0.0]
        expected_constrained = np.array([0.5 / np.sqrt(2), 0.5 / np.sqrt(2), 0.0])
        actual_params = callback._flatten_params(model)
        np.testing.assert_allclose(actual_params, expected_constrained, rtol=1e-5)

        # Also check that pressure is non-zero (since we projected)
        assert metrics["othismos/pressure"] > 0

        # Now test with apply_constraints=False (default)
        callback2 = OthismosTorchCallback(constraints=constraints, apply_constraints=False)
        callback2.pre_step(model, None, None)
        # Reset model parameters to the same outside value
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 2.0]]))
            model.bias.copy_(torch.tensor([0.0]))
        callback2.post_step(model)
        # Parameters should remain unchanged (still the outside value) because we didn't write back
        actual_params2 = callback2._flatten_params(model)
        np.testing.assert_allclose(actual_params2, [2.0, 2.0, 0.0], rtol=1e-5)


class TestPublicAPIContract:
    """Regression for Bug #7: Public API contract — no phantom class names.

    The original beta-review claimed that __init__.py exported `Othismos`
    but the actual class was named `OthismosEngine`. This test pins the
    actual public API so we catch any future drift.
    """

    def test_no_phantom_othismosengine_class(self):
        """There should be NO class named OthismosEngine in the package."""
        import othismos
        # OthismosEngine should never be importable
        assert not hasattr(othismos, "OthismosEngine"), (
            "othismos.OthismosEngine should NOT exist — no such class was ever defined"
        )
        # It also shouldn't be importable from any submodule
        from othismos import pressure, phases, ecology, controller
        for module in (othismos, pressure, phases, ecology, controller):
            assert not hasattr(module, "OthismosEngine"), (
                f"{module.__name__}.OthismosEngine should NOT exist"
            )

    def test_top_level_othismos_alias_does_not_exist(self):
        """There is no top-level Othismos class — verify the import error."""
        import pytest
        with pytest.raises(ImportError):
            from othismos import Othismos  # noqa: F401

    def test_actual_main_classes_are_importable(self):
        """Verify the actual top-level classes ARE importable."""
        # These are the real public API
        from othismos import (
            OthismosConfig,
            PressureGauge,
            PressureController,
            Reef,
            OthismosTorchCallback,
            OthismosTrainerCallback,
            PhaseClassifier,
            PopcornDiagnostic,
            save_history,
            load_history,
        )
        assert OthismosConfig is not None
        assert PressureGauge is not None
        assert PressureController is not None

