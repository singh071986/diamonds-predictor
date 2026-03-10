# Product Requirements Document (PRD)

## Product
Diamonds Predictor Application

## Version
1.0 (Draft aligned with current repository state)

## 1. Product Goals
1. Provide a single Python-based mini platform to analyze, preprocess, and model the diamonds dataset.
2. Enable three ML outcomes from the same feature set:
   - Classification: predict diamond quality class (primary target: `cut`; extensible to `clarity`).
   - Regression: predict diamond price.
   - Clustering: segment diamonds into meaningful groups for customer/product strategy.
3. Support reproducible experimentation with consistent APIs (`fit`, `predict`, `score`) and saved outputs (plots, cleaned data).
4. Deliver educational and portfolio-quality project structure suitable for GitHub submission.

## 2. Target Users
1. Bootcamp learner implementing guided capstone requirements.
2. Reviewer/instructor validating project completeness and ML workflow quality.
3. Future collaborator who wants to run, extend, and compare models.

## 3. User Stories
1. As a learner, I want to load CSV data and inspect shape/types so I can understand raw input quality.
2. As a learner, I want preprocessing (drop columns, encode categoricals, shuffle, sample) so my models can train on clean numeric inputs.
3. As a learner, I want to generate EDA plots (correlation, histograms, boxplots, pairplot) and save them for reporting.
4. As a learner, I want to train multiple classifiers and compare accuracy/confusion matrix so I can select the best model.
5. As a learner, I want to train multiple regressors and compare R2/MSE/RMSE/MAE so I can estimate diamond prices reliably.
6. As a learner, I want to run clustering (KMeans/Agglomerative/MeanShift) and inspect elbow/inertia so I can segment diamonds.
7. As a reviewer, I want deterministic tests for core APIs so I can verify baseline functionality quickly.

## 4. Functional Requirements

### 4.1 Analyzer Module
1. Read dataset from CSV path.
2. Show head/info/describe summaries.
3. Drop selected columns (including `Unnamed: 0` when present).
4. Encode categorical feature columns and optional label column.
5. Shuffle rows with fixed `random_state` support.
6. Sample data by reduction factor `0.0 < r <= 1.0`.
7. Save cleaned dataset to CSV.
8. Plot and optionally save:
   - Correlation matrix
   - Categorical histograms
   - Numerical histograms
   - Boxplots
   - Pairplot

### 4.2 Classifier Module
1. Unified `fit(model_name, X_train, y_train, **params)` for `knn`, `decision_tree`, `random_forest`, `svc`, `ann`.
2. `predict(model_name, X)` returning label predictions.
3. `score(model_name, X, y, metric)` supporting at least `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`.
4. Confusion matrix plotting utility that returns matrix values.
5. Hyperparameter search helper for KNN `k` values and best score output.

### 4.3 Regressor Module
1. Unified `fit(model_name, X_train, y_train, **params)` for `knn`, `decision_tree`, `random_forest`, `svr`, `ann`.
2. `predict(model_name, X)` returning numeric predictions.
3. `score(model_name, X, y)` returning dict with `MSE`, `RMSE`, `MAE`, `R2`.
4. Hyperparameter helper for KNN and tree/random forest options.

### 4.4 Clustering Module
1. `fit(model_name, X, **params)` for `kmeans`, `agglomerative`, `mean_shift`.
2. `predict(model_name, X_new)` where model supports prediction.
3. KMeans inertia retrieval and elbow-curve plotting utility.
4. Optional silhouette score reporting for `k >= 2`.

### 4.5 Main/Pipeline
1. Executable script to run representative flows:
   - Analyze + preprocess + save
   - Train/evaluate at least one classifier
   - Train/evaluate at least one regressor
   - Run clustering and plot elbow
2. Clear console output of selected model metrics.

## 5. Non-Functional Requirements
1. Compatibility: Python 3.10+ on macOS/Windows/Linux.
2. Maintainability: modular files (`Analyzer.py`, `Classifier.py`, `Regressor.py`, `Clustering.py`, `main.py`).
3. Reproducibility: random seeds for train/test split and shuffling.
4. Performance: core smoke flow completes on diamonds dataset within typical laptop constraints (<10 minutes excluding heavy ANN tuning).
5. Robustness: meaningful errors for invalid model names/metrics/invalid sample factors.
6. Documentation: clear README instructions and examples.
7. Testability: unit tests for core API behaviors and metric outputs.

## 6. Acceptance Criteria

### Phase 1 Acceptance (A1-A5)
- A1: Analyzer can load CSV, drop `Unnamed: 0`, and output cleaned dataframe without missing values for test fixture.
- A2: Analyzer `sample(reduction_factor)` returns approximately expected row count and validates factor bounds.
- A3: Classifier supports unified `fit/predict/score` for at least `knn` and `decision_tree` and returns accuracy in range `[0,1]`.
- A4: Regressor supports unified `fit/predict/score` for at least `knn` and `decision_tree` and returns keys `MSE`, `RMSE`, `MAE`, `R2`.
- A5: Clustering supports unified `fit` for `kmeans`, stores inertia, and returns cluster labels with valid length.

### Full Product Acceptance
1. All required estimators execute without runtime errors on cleaned diamonds data.
2. Classification baseline accuracy for `cut` is >= 0.50 on at least one model.
3. Regression baseline R2 for `price` is >= 0.90 on at least one model after tuning.
4. KMeans elbow chart is generated and inertia values are retrievable.
5. Confusion matrix plotting works for trained classifier models.
6. README includes setup, run instructions, and module usage.

## 7. Milestones
1. M1 - Foundation and Contracts (Phase 1)
   - Introduce unified APIs and baseline tests (A1-A5).
2. M2 - Analyzer + Visualization Completion
   - Complete all required preprocessing and plotting save options.
3. M3 - Classification Completion
   - Add all classifier estimators, confusion matrix, and simple tuning helpers.
4. M4 - Regression Completion
   - Add all regressors, metric dictionary scoring, and baseline tuning outputs.
5. M5 - Clustering Completion
   - Add all clustering estimators, inertia helper, elbow and cluster evaluation.
6. M6 - Integration and Documentation
   - End-to-end main pipeline, README updates, final validation.

## 8. Risks
1. TensorFlow/Keras installation/version issues across machines.
2. Long training time for ANN models on CPU-only environments.
3. Data leakage risk if preprocessing/scaling is done before train/test split incorrectly.
4. Inconsistent label encoding between train/test when not persisted.
5. Plotting calls can fail in headless environments unless save-only mode is available.

## 9. Assumptions
1. `diamonds.csv` is available in repository root.
2. Primary classification target is `cut`; `clarity` can be supported as extension.
3. Project is educational first, production hardening second.
4. User can install required dependencies from `requirements.txt`.
5. Tests use lightweight synthetic/fixture subsets for speed.

## 10. Out of Scope (for initial completion)
1. Model serving API (Flask/FastAPI).
2. Frontend web application UI.
3. Automated CI/CD deployment.
4. Advanced hyperparameter optimization frameworks (Optuna, Ray Tune).

## 11. Requirements-to-Code Traceability

The table below maps project and screenshot requirements to implemented code and validation tests.

| Requirement Area | Required Capability | Implemented In | Validation |
|---|---|---|---|
| Analyzer | `read_dataset`, `describe` | `Analyzer.py` (`read_dataset`, `describe`) | `tests/test_phase1_acceptance.py` |
| Analyzer | `drop_missing_data`, `drop_columns` | `Analyzer.py` (`drop_missing_data`, `drop_columns`) | `tests/test_phase1_acceptance.py` (A1) |
| Analyzer | `encode_features`, `encode_label` | `Analyzer.py` (`encode_features`, `encode_label`) | API-level verification and pipeline usage in `main.py` |
| Analyzer | `Shuffle`, `Sample` | `Analyzer.py` (`shuffle`, alias `Shuffle`, `sample`) | `tests/test_phase1_acceptance.py` (A2) |
| Analyzer | Correlation matrix plot | `Analyzer.py` (`plot_correlation_matrix`, alias `plot_correlationMatrix`) | `tests/test_phase2_plotting.py` |
| Analyzer | Pair plot | `Analyzer.py` (`plot_pairPlot`) | artifact generation in `main.py` |
| Analyzer | Numerical histogram plot | `Analyzer.py` (`plot_histograms_numerical`, alias `Plot_histograms_numerical`) | `tests/test_phase2_plotting.py` |
| Analyzer | Categorical histogram plot | `Analyzer.py` (`plot_histograms_categorical`, alias `Plot_histograms_categorical`) | artifact generation in `main.py` |
| Analyzer | Box plot | `Analyzer.py` (`plot_boxPlot`, alias `Plot_boxPlot`) | `tests/test_phase2_plotting.py` |
| Classifier | `fit`, `predict`, `score` contract | `Classifier.py` | `tests/test_phase1_acceptance.py` (A3) |
| Classifier | Estimators: Logistic Regression, KNN, Decision Tree, Random Forest, SVC, ANN | `Classifier.py` (`_build_model`, `train_*`) | `tests/test_phase3_5_completion.py` |
| Classifier | Accuracy metric (`accuracy_score`) | `Classifier.py` (`score`) | `tests/test_phase3_5_completion.py` |
| Classifier | Confusion matrix output | `Classifier.py` (`plot_confusion_matrix`) | pipeline run in `main.py` + tests |
| Regressor | `fit`, `predict`, `score` contract | `Regressor.py` | `tests/test_phase1_acceptance.py` (A4) |
| Regressor | Estimators: Linear Regression, KNN, Decision Tree, Random Forest, SVR, ANN | `Regressor.py` (`_build_model`, `train_*`) | `tests/test_phase3_5_completion.py` |
| Regressor | Metrics: `R2`, `MSE`, `RMSE`, `MAE` | `Regressor.py` (`score`) | `tests/test_phase1_acceptance.py` + `tests/test_phase3_5_completion.py` |
| Regressor | Tuning helpers (KNN/tree/forest) | `Regressor.py` (`tune_knn`, `tune_decision_tree`, `tune_random_forest`) | used by `main.py` regression flow |
| Clustering | `fit`, `predict` contract | `Clustering.py` | `tests/test_phase1_acceptance.py` (A5) |
| Clustering | Estimators: KMeans, Agglomerative, Mean Shift | `Clustering.py` (`fit`) | `tests/test_phase3_5_completion.py` |
| Clustering | KMeans inertia + elbow + silhouette | `Clustering.py` (`get_kmeans_inertia`, `elbow_curve`, `elbow_with_silhouette`) | `tests/test_phase3_5_completion.py` |
| Pipeline | End-to-end analyze/classify/regress/cluster flow | `main.py` (`run_analyzer`, `run_classification`, `run_regression`, `run_clustering`) | `tests/test_phase3_5_completion.py` |
| Documentation | Setup, module behavior, and run instructions | `README.md`, `docs/ExecutionPlan.md`, `docs/PRD.md` | manual review + runnable commands |
