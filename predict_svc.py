import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd


def load_model_bundle(model_path):
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and 'model' in bundle:
        return bundle
    if isinstance(bundle, dict) and bundle.get('model_type') == 'ann':
        return bundle
    return {
        'model': bundle,
        'feature_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'model_type': 'sklearn',
    }


def coerce_value(value, is_numeric):
    if is_numeric:
        return float(value)
    return str(value)


def gather_features_interactive(feature_columns, numeric_columns):
    feature_dict = {}
    numeric_set = set(numeric_columns)
    print('Enter feature values for prediction:')
    for col in feature_columns:
        raw = input(f'{col}: ').strip()
        if raw == '':
            raise ValueError(f'Missing value for feature: {col}')
        feature_dict[col] = coerce_value(raw, col in numeric_set)
    return feature_dict


def main():
    parser = argparse.ArgumentParser(description='Predict diamond cut using saved SVC or ANN model bundle.')
    parser.add_argument(
        '--model-path',
        default='artifacts/models/svc_cut_model.joblib',
        help='Path to saved SVC model bundle.',
    )
    parser.add_argument(
        '--json',
        help='Input features as JSON string. Example: --json "{\"carat\":0.7,\"color\":\"E\",...}"',
    )
    args = parser.parse_args()

    bundle = load_model_bundle(args.model_path)
    model_type = bundle.get('model_type', 'sklearn')
    model = bundle.get('model')
    feature_columns = bundle.get('feature_columns', [])
    numeric_columns = bundle.get('numeric_columns', [])

    if args.json:
        payload = json.loads(args.json)
        if feature_columns:
            missing = [c for c in feature_columns if c not in payload]
            if missing:
                raise ValueError(f'Missing fields in JSON input: {missing}')
            ordered_payload = {c: payload[c] for c in feature_columns}
        else:
            ordered_payload = payload
    else:
        if not feature_columns:
            raise ValueError('Model bundle does not include feature metadata for interactive mode.')
        ordered_payload = gather_features_interactive(feature_columns, numeric_columns)

    input_df = pd.DataFrame([ordered_payload])
    if model_type == 'ann':
        keras_model_path = bundle.get('keras_model_path')
        preprocessor = bundle.get('preprocessor')
        label_encoder = bundle.get('label_encoder')
        if not keras_model_path or preprocessor is None or label_encoder is None:
            raise ValueError('ANN model bundle is missing required keys: keras_model_path/preprocessor/label_encoder')

        # Import TensorFlow only when ANN inference is requested.
        from tensorflow.keras.models import load_model

        resolved_keras_path = (
            keras_model_path
            if os.path.isabs(keras_model_path)
            else os.path.join(os.path.dirname(args.model_path), os.path.basename(keras_model_path))
        )
        ann_model = load_model(resolved_keras_path)
        transformed = preprocessor.transform(input_df)
        probs = ann_model.predict(transformed, verbose=0)
        pred_index = int(np.argmax(probs, axis=1)[0])
        prediction = label_encoder.inverse_transform([pred_index])[0]
    else:
        if model is None:
            raise ValueError('Model bundle does not contain a sklearn model under key "model".')
        prediction = model.predict(input_df)[0]

    print(f'Predicted cut: {prediction}')


if __name__ == '__main__':
    main()
