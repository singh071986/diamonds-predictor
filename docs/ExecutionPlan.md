# Execution Roadmap

Source of truth: docs/PRD.md

## Delivery Strategy
Implement in phases with test-first checkpoints. Each phase closes a milestone and leaves repository in a runnable state.

## Phase 1 - Foundation and API Contracts
Goal: stabilize shared interfaces and pass A1-A5.

Tasks:
1. Refactor Analyzer with deterministic preprocess, `drop_columns`, `sample(reduction_factor)`, and robust validations.
2. Refactor Classifier to unified API: `fit/predict/score` with model registry.
3. Refactor Regressor to unified API: `fit/predict/score` returning metric dict.
4. Refactor Clustering to unified API with KMeans inertia support.
5. Add pytest suite covering acceptance criteria A1-A5.

Dependencies:
1. `pandas`, `numpy`, `scikit-learn`, `pytest` available.

Exit criteria:
1. All tests for A1-A5 pass.
2. `main.py` still runnable (smoke-level).

## Phase 2 - Analyzer Visualization Completion
Goal: complete and standardize plotting capabilities with save support.

Tasks:
1. Add `save_path` + `show` toggles for all plot functions.
2. Ensure categorical and numerical plotting functions accept column lists.
3. Add tests for no-crash plot generation in non-interactive backend.

Dependencies:
1. Matplotlib/seaborn backend stability.

Exit criteria:
1. Plot files generated successfully in tests.

## Phase 3 - Classification Completion
Goal: implement all required classifier estimators and confusion matrix helper.

Tasks:
1. Add `random_forest`, `svc`, and optional `ann` support via unified `fit`.
2. Add score metrics selection by string names.
3. Add confusion matrix plotting + return matrix.
4. Add simple KNN search utility for best `k`.

Dependencies:
1. Phase 1 API complete.
2. Optional TensorFlow if ANN enabled.

Exit criteria:
1. At least one classifier reaches >= 0.50 accuracy on `cut`.

## Phase 4 - Regression Completion
Goal: implement all required regressors and scoring metrics.

Tasks:
1. Add `random_forest`, `svr`, optional `ann` via unified `fit`.
2. Guarantee `score` outputs `MSE`, `RMSE`, `MAE`, `R2`.
3. Add lightweight tuning helpers for KNN/tree/random forest.

Dependencies:
1. Phase 1 API complete.

Exit criteria:
1. At least one regressor reaches >= 0.90 R2 on `price` after tuning.

## Phase 5 - Clustering Completion
Goal: finalize clustering features and diagnostics.

Tasks:
1. Add `agglomerative` and `mean_shift` in unified `fit`.
2. Add elbow helper returning inertias and optional silhouette scores.
3. Add tests for cluster label lengths and inertia retrieval.

Dependencies:
1. Phase 1 `Clustering` API complete.

Exit criteria:
1. KMeans elbow output and inertia retrieval validated.

## Phase 6 - Integration and Documentation
Goal: align end-to-end pipeline and docs with final behavior.

Tasks:
1. Update `main.py` to use unified APIs.
2. Update `README.md` with module usage and reproducible commands.
3. Add final project checklist mapping PRD acceptance criteria.

Dependencies:
1. Phases 1-5 complete.

Exit criteria:
1. Clean run on diamonds dataset with outputs and metrics.
2. Documentation fully aligned.

## Test Strategy
1. Unit tests
   - API contracts for Analyzer/Classifier/Regressor/Clustering.
   - Validation errors (invalid model names, bad sample factors).
2. Integration smoke tests
   - Small data slice: preprocess -> classify -> regress -> cluster.
3. Performance sanity tests
   - Ensure default non-ANN tests finish quickly.
4. Optional ANN tests
   - Mark as slow/optional due to environment variance.

## Task Dependency Graph (High Level)
1. Phase 1 -> Phase 2 -> Phase 6
2. Phase 1 -> Phase 3 -> Phase 6
3. Phase 1 -> Phase 4 -> Phase 6
4. Phase 1 -> Phase 5 -> Phase 6

## Current Status
1. Phase 1 completed (A1-A5 passed).
2. Phase 2 partially completed (plot save/show support and tests).
3. Phase 3-5 core implementation completed in code with unified APIs and diagnostics.
4. Phase 6 integration and README alignment completed.

## Immediate Next Action
1. Validate end-to-end with automated tests and confirm acceptance thresholds.
2. Close any remaining warnings or non-blocking gaps.
