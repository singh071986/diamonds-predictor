# Diamonds Predictor Application

Mini machine learning platform for diamonds analysis, classification, regression, and clustering.

## What is implemented

1. Analyzer
- CSV loading and dataset inspection.
- Preprocessing: missing-value removal, automatic dropping of `Unnamed*` columns, encoding, scaling, shuffle, and sampling.
- Visualizations with save/show controls:
	correlation matrix, categorical histograms, numerical histograms, pairplot, and boxplot.
- Plotting methods automatically ignore `Unnamed*` columns, even when raw CSV is loaded.

2. Classifier
- Unified API: `fit`, `predict`, `score`.
- Supported estimators: `knn`, `decision_tree`, `random_forest`, `svc`, optional `ann`.
- Metric support: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`.
- KNN tuning helper and confusion matrix plotting with return value.

3. Regressor
- Unified API: `fit`, `predict`, `score`.
- Supported estimators: `knn`, `decision_tree`, `random_forest`, `svr`, optional `ann`.
- Score output: `MSE`, `RMSE`, `MAE`, `R2`.
- Tuning helpers for KNN, decision tree criterion, and random forest params.

4. Clustering
- Unified API: `fit`, `predict` (when supported by model).
- Supported estimators: `kmeans`, `agglomerative`, `mean_shift`.
- Diagnostics: KMeans inertia retrieval, elbow curve, elbow + silhouette summary.

5. Pipeline
- `main.py` runs full workflow and writes plot artifacts to `artifacts/`.
- Pipeline uses a shared clean-first step (`prepare_clean_data`) before classification, regression, and clustering.

## Project structure

```text
dimond_project/
|-- Analyzer.py
|-- Classifier.py
|-- Regressor.py
|-- Clustering.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- docs/
|   |-- PRD.md
|   `-- ExecutionPlan.md
`-- tests/
```

## Run

### Recommended setup (script)

From project root, run:

```bash
conda deactivate
deactivate 2>/dev/null || true
bash scripts/setup_conda.sh
```

Optional custom env name and Python version:

```bash
bash scripts/setup_conda.sh my_env 3.10
```

If you use a custom name (for example `my_env`), use that same name in all
`conda run -n ...` commands below.

### Manual fallback (only if needed)

```bash
conda deactivate
deactivate 2>/dev/null || true
conda create -n diamond_env python=3.10 -y
conda run -n diamond_env python -m pip install -r requirements.txt
conda run -n diamond_env python -c "import tensorflow as tf; print(tf.__version__)"
```

1. Ensure `diamonds.csv` exists in project root.

2. Execute pipeline

```bash
conda run -n diamond_env python main.py
```

If you created `my_env` instead, run:

```bash
conda run -n my_env python main.py
```

Optional sanity check for package import in a specific env:

```bash
conda run -n diamond_env python -c "import pandas as pd; print(pd.__version__)"
```

Artifacts are saved under `artifacts/`.

## Tests

Run all tests:

```bash
conda run -n diamond_env python -m unittest discover -s tests -p "test_*.py"
```

If you created `my_env` instead, run:

```bash
conda run -n my_env python -m unittest discover -s tests -p "test_*.py"
```

## Notes

1. ANN models require TensorFlow/Keras and can be slower on CPU-only machines.
2. Exact metrics vary by split and sample size.
3. Base install includes TensorFlow/Keras, so use Python 3.10-3.12 for best compatibility.
