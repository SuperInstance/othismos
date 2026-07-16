# Changelog

All notable changes to othismos will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-16

### Fixed
- **CRITICAL #1**: `l2_constraint` now correctly handles N-dimensional `theta`.
  Default `center` is `np.zeros_like(theta)` instead of `np.zeros(1)`, so the
  projection math is shape-correct for 1-D, 2-D, and N-D parameter tensors.
  (Regression test: `test_l2_constraint_n_dimensional_theta`)
- **CRITICAL #2**: `PhaseClassifier` now uses explicit `is not None` checks for
  `crisis_threshold` and `expansion_floor`. Previously `crisis_threshold=0.0`
  was silently ignored because `0.0 or default` evaluates to the default.
  (Regression tests: `test_crisis_threshold_zero_is_respected`,
  `test_expansion_floor_zero_is_respected`)
- **CRITICAL #3**: `save_history` and `save_diagnostic` now serialize numpy
  scalars (`np.int64`, `np.float64`, `np.bool_`) and arrays via a dedicated
  `NumpyEncoder`. No more `TypeError: Object of type int64 is not JSON
  serializable`. (Regression test: `test_numpy_scalar_serialization`)
- **CRITICAL #4**: `Reef.tick()` no longer mutates `self._deposits` during
  iteration. It now iterates over `list(self._deposits.items())`, eliminating
  the `RuntimeError: dictionary changed size during iteration` on large reefs.
  (Regression tests: `test_tick_with_mass_erosion_no_runtimeerror`,
  `test_tick_interleaved_add_and_erosion`)
- **CRITICAL #5**: `OthismosTorchCallback` now supports an `apply_constraints`
  flag. When `True`, the projected parameters are written back to the model
  (with the proper tensor data flow). Default is `False` (observatory-only).
  (Regression test: `test_apply_constraints_flag`)
- **CRITICAL #6**: `PressureGauge` supports a `store_vectors=False` mode that
  stores only scalar norms (pressure, desired_norm, actual_norm,
  violation_norm), reducing per-step memory from O(n_params) to O(1). This
  was previously verified in commit `bb1fa47`. (Regression tests:
  `test_store_vectors_flag`, `test_store_vectors_memory_efficiency`)
- **CRITICAL #7** (N/A): The reviewer claim that `__init__.py` exported a
  phantom `Othismos` class whose real name was `OthismosEngine` was
  investigated and found to be **incorrect** — no such class exists.
  Verified via `test_no_phantom_othismosengine_class`.

### Changed
- `pressure_summary` now wraps all returned scalars through `_ndarray_to_list`
  so the dict is JSON-serializable out of the box.
- `OthismosTorchCallback.__init__` documents the `apply_constraints` flag and
  the torch-optional nature of the integration (torch is lazy-imported inside
  methods, so the module loads cleanly without torch).
- README "Repository structure" section updated to reflect actual filenames
  (`04_THE_POPCORN_DIAGNOSTIC.md` not `04_POPCORN.md`, etc.). Six broken
  filename references corrected.

### Documented
- **MEDIUM #8**: The sequential constraint projection in `compute_othismos`
  does not perform fixed-point iteration. This is documented in a comment
  with guidance on when it matters and how to work around it. Future work:
  add an `iterative=True` flag.
- **MEDIUM #9**: torch is verified to be a fully optional dependency.
  `integrations.py` and `llm.py` import torch lazily inside methods. The
  `OthismosTorchCallback` class can be instantiated without torch installed;
  only the methods that actually use torch require it.
- **MEDIUM #10**: README now references all existing docs by their correct
  filenames.

### Tests
- Total tests: **135** (up from 122 at v0.3.0)
- +13 regression tests for the 7 critical bugs above
- All tests passing

## [0.3.0] - 2026-07-14

### Added
- Protocol abstractions (`FeasibilityFn`, `ProjectionFn`, `NormalFn`,
  `DistanceFn`, `ConstraintLike`)
- `OthismosConfig` dataclass with YAML/dict serialization and stack builders
- Visualization module (`plot_pressure`, `plot_molt_cycle`,
  `plot_constraint_profile`, `plot_diagnostic_timeline`) — requires
  `matplotlib`
- Pandas export module (`gauge_to_dataframe`, `tracker_to_dataframe`,
  `reef_to_dataframe`) — requires `pandas`
- `LLMPressureAnalyzer` module for context-window pressure in language models
- 122 tests across 10 test files

### Fixed (early v0.3.0 cycle, prior to this changelog file)
- `l2_constraint` default center broadcasting (initial fix in commit `f1a28ac`)
- `PhaseClassifier` `or` vs `is None` (commit `f1a28ac`)
- `save_history` numpy scalar JSON serialization (commit `f1a28ac`)
- `Reef.tick()` dict mutation safety (verified safe in commit `f1a28ac`)
- `PressureGauge` OOM via `store_vectors=False` (commit `bb1fa47`)

## [0.2.0] - 2026-07-13

### Added
- PyTorch integration (`OthismosTorchCallback`)
- HuggingFace Trainer integration (`OthismosTrainerCallback`)
- Serialization module (`save_history`, `load_history`, `save_diagnostic`,
  `export_metrics_csv`, `pressure_summary`)
- 70 tests

## [0.1.0] - 2026-07-12

### Added
- Initial release
- Core pressure measurement (`PressureMeasurement`, `PressureGauge`,
  `compute_othismos`)
- Constraint types (`l2_constraint`, `box_constraint`)
- Phase classification (`PhaseClassifier`, `MoltCycleTracker`)
- Popcorn diagnostic (`PopcornDiagnostic`, `DiagnosticResult`)
- Reef ecology (`Reef`, `Deposit`, `ReefLayer`, `Reefquake`)