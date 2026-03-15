import argparse
import json
import joblib
import pandas as pd


def load_model_bundle(model_path):
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and 'model' in bundle:
        return bundle
    return {
        'model': bundle,
        'feature_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
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
    parser = argparse.ArgumentParser(description='Predict diamond cut using saved SVC model.')
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
    model = bundle['model']
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
    prediction = model.predict(input_df)[0]
    print(f'Predicted cut: {prediction}')


if __name__ == '__main__':
    main()
