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

## Test User-Provided Input (Saved SVC Model)

After running the pipeline once, an SVC model bundle is saved under:

```text
artifacts/models/svc_cut_model.joblib
```

You can test custom user input using `predict_svc.py` in two ways.

1. JSON input (single command):

```bash
conda run -n diamond_env python predict_svc.py \
	--model-path artifacts/models/svc_cut_model.joblib \
	--json '{"carat":0.7,"color":"E","clarity":"VS2","depth":61.8,"table":57.0,"price":3400,"x":5.7,"y":5.72,"z":3.53}'
```

If you use `my_env`, run:

```bash
conda run -n my_env python predict_svc.py \
	--model-path artifacts/models/svc_cut_model.joblib \
	--json '{"carat":0.7,"color":"E","clarity":"VS2","depth":61.8,"table":57.0,"price":3400,"x":5.7,"y":5.72,"z":3.53}'
```

2. Interactive input (prompted field-by-field):

```bash
conda run -n diamond_env python predict_svc.py --model-path artifacts/models/svc_cut_model.joblib
```

If you use `my_env`, run:

```bash
conda run -n my_env python predict_svc.py --model-path artifacts/models/svc_cut_model.joblib
```

The script prints the predicted class, for example: `Predicted cut: Ideal`.

## Test User Input via Node.js (Using Saved SVC Model)

`predict_svc_node.js` is a Node wrapper that calls `predict_svc.py`, so Node can use the saved scikit-learn SVC model.

1. Ensure Node.js is installed:

```bash
node -v
```

2. Run prediction with JSON input:

```bash
node predict_svc_node.js \
	--model-path artifacts/models/svc_cut_model.joblib \
	--json '{"carat":0.7,"color":"E","clarity":"VS2","depth":61.8,"table":57.0,"price":3400,"x":5.7,"y":5.72,"z":3.53}'
```

3. If needed, point Node to a specific Python executable path:

```bash
PYTHON_CMD=/opt/anaconda3/bin/python node predict_svc_node.js \
	--model-path artifacts/models/svc_cut_model.joblib \
	--json '{"carat":0.7,"color":"E","clarity":"VS2","depth":61.8,"table":57.0,"price":3400,"x":5.7,"y":5.72,"z":3.53}'
```

Expected output:

```text
Prediction: Ideal
```

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



python predict_svc.py --model-path artifacts/models/svc_cut_model_test.joblib --json '{"carat":0.7,"color":"E","clarity":"VS2","depth":61.8,"table":57.0,"price":3400,"x":5.7,"y":5.72,"z":3.53}'



node predict_svc_node.js --model-path artifacts/models/svc_cut_model_test.joblib --json '{"carat":0.7,"color":"E","clarity":"VS2","depth":61.8,"table":57.0,"price":3400,"x":5.7,"y":5.72,"z":3.53}'
