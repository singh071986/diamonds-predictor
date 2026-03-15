from Analyzer import Analyzer
from Classifier import Classifier
from Regressor import Regressor
# from Clustering import Clustering
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


def prompt_text(message, default=''):
    """Read user input, falling back to default when stdin is unavailable."""
    try:
        value = input(message).strip()
    except EOFError:
        value = ''

    if value:
        return value

    if default:
        print(f'No interactive input detected. Using default: {default}')
    return default


def prompt_positive_int(message, default):
    """Read a positive integer with fallback default."""
    while True:
        value = prompt_text(message, default=str(default))
        if value.isdigit() and int(value) > 0:
            return int(value)
        print('Invalid input. Please enter a positive integer.')


def prepare_clean_data(df):
    """Return a consistently cleaned dataframe for all downstream tasks.
    Cleaning rules:
    1) Drop auto-generated index columns (e.g., Unnamed: 0).
    2) Drop rows with missing values.
    """
    cleaned = df.copy()
    unnamed_cols = [c for c in cleaned.columns if str(c).lower().startswith('unnamed')]
    if unnamed_cols:
        cleaned = cleaned.drop(columns=unnamed_cols)
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def encode_categoricals(df):
    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns
    return pd.get_dummies(df, columns=categorical_columns, drop_first=False, dtype=float)


def plot_classification_metrics(metrics, save_path='artifacts/classification_model_comparison.png', show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    base, ext = os.path.splitext(save_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_save_path = f'{base}_{timestamp}{ext or ".png"}'
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.ylabel('Accuracy')
    plt.title('Classification Model Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha='right')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(timestamped_save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_regression_metrics(metrics, save_path='artifacts/regression_model_comparison.png', show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    models = list(metrics.keys())
    metric_names = ['R2', 'MAE', 'RMSE', 'MSE']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        values = [metrics[model][metric_name] for model in models]
        bars = axes[idx].bar(models, values)
        axes[idx].set_title(metric_name)
        axes[idx].tick_params(axis='x', rotation=20)

        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, value, f'{value:.3f}', ha='center', va='bottom')

    fig.suptitle('Regression Model Comparison', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_analyzer(data_path='diamonds.csv', output_path='cleaned_diamonds.csv'):
    analyzer = Analyzer(data_path)
    analyzer.show_info()

    # Plot category distributions before preprocessing converts categories to numeric values.
    analyzer.data = prepare_clean_data(analyzer.data)
    analyzer.plot_histograms_categorical(save_dir='artifacts/categorical_histograms', show=False)

    analyzer.preprocess_data()
    # Enforce shared cleaning policy before artifacts and export.
    analyzer.data = prepare_clean_data(analyzer.data)
    analyzer.save_cleaned_data(output_path)
    analyzer.plot_correlation_matrix(save_path='artifacts/correlation_matrix.png', show=False)
    analyzer.plot_histograms_numerical(save_path='artifacts/histograms_numerical.png', show=False)
    analyzer.plot_pairPlot(save_path='artifacts/pairplot.png', show=False)
    analyzer.plot_boxPlot('price', save_path='artifacts/boxplot_price.png', show=False)
    return analyzer.data


def run_classification(data, ann_config=None):
    cleaned = prepare_clean_data(data)
    features = cleaned.drop(columns=['cut'])
    categorical_columns = features.select_dtypes(include=['object', 'string', 'category']).columns
    numeric_columns = features.select_dtypes(include=['number']).columns

    if ann_config is None:
        ann_config = {
            'epochs': 20,
            'batch_size': 32,
            'hidden_1': 64,
            'hidden_2': 32,
        }

    # Build final feature matrix X by combining:
    # 1) one-hot encoded categorical columns and
    # 2) scaled numeric columns.
    if len(categorical_columns) > 0:
        encoded_categorical = pd.get_dummies(
            features[categorical_columns], drop_first=False, dtype=float
        )

    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        scaled_numeric = pd.DataFrame(
            scaler.fit_transform(features[numeric_columns]),
            columns=numeric_columns,
            index=features.index,
        )


    X = pd.concat([scaled_numeric, encoded_categorical], axis=1)
    classification_data = pd.concat([X, cleaned['cut'].reset_index(drop=True)], axis=1)

    classifier = Classifier(classification_data, target='cut')
    X_train, X_test, y_train, y_test = train_test_split(
        classifier.X, classifier.y, test_size=0.2, random_state=42, stratify=classifier.y
    )

    classifier.fit('logistic_regression', X_train, y_train, max_iter=1000)
    best_k, _, _ = classifier.tune_knn()
    classifier.fit('knn', X_train, y_train, k=best_k)
    classifier.fit('decision_tree', X_train, y_train, criterion='gini')
    classifier.fit('random_forest', X_train, y_train, criterion='gini', n_estimators=500)
    classifier.fit('svc', X_train, y_train, kernel='rbf', C=2.0)

    classifier.fit(
        'ann',
        X_train,
        y_train,
        input_dim=X_train.shape[1],
        n_classes=len(pd.Series(y_train).unique()),
        epochs=ann_config['epochs'],
        batch_size=ann_config['batch_size'],
        hidden_1=ann_config['hidden_1'],
        hidden_2=ann_config['hidden_2'],
    )

    metrics = {}
    models_to_score = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svc', 'ann']

    for model in models_to_score:
        metrics[model] = classifier.score(model, X_test, y_test, metric='accuracy')

    classifier.plot_confusion_matrix('random_forest', X_test, y_test, save_path='artifacts/classifier_confusion_matrix.png', show=False)
    return metrics


def run_regression(data):
    cleaned = prepare_clean_data(data)
    encoded = encode_categoricals(cleaned)
    regressor = Regressor(encoded, target='price')
    X_train, X_test, y_train, y_test = train_test_split(
        regressor.X, regressor.y, test_size=0.2, random_state=42
    )

    regressor.fit('linear_regression', X_train, y_train)
    best_k, _, _ = regressor.tune_knn()
    best_tree_criterion, _, _ = regressor.tune_decision_tree()
    best_rf_params, _, _ = regressor.tune_random_forest()

    regressor.fit('knn', X_train, y_train, k=best_k)
    regressor.fit('decision_tree', X_train, y_train, criterion=best_tree_criterion)
    regressor.fit(
        'random_forest',
        X_train,
        y_train,
        n_estimators=best_rf_params['n_estimators'],
        criterion=best_rf_params['criterion'],
    )
    regressor.fit('svr', X_train, y_train, kernel='rbf', C=5.0)

    metrics = {}
    for model in ['linear_regression', 'knn', 'decision_tree', 'random_forest', 'svr']:
        metrics[model] = regressor.score(model, X_test, y_test)
    return metrics


def train_and_save_svc_model(data, model_path='artifacts/models/svc_cut_model.joblib'):
    cleaned = prepare_clean_data(data)
    X = cleaned.drop(columns=['cut'])
    y = cleaned['cut']

    categorical_columns = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numeric_columns = X.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_columns),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ]
    )

    svc_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', SVC(kernel='rbf', C=2.0, gamma='scale')),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    svc_pipeline.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, svc_pipeline.predict(X_test))

    # Refit on full selected data before saving for inference usage.
    svc_pipeline.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            'model': svc_pipeline,
            'feature_columns': X.columns.tolist(),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'holdout_accuracy': float(test_accuracy),
        },
        model_path,
    )
    return model_path, test_accuracy


# def run_clustering(data):
#     cleaned = prepare_clean_data(data)
#     encoded = encode_categoricals(cleaned)
#     features = encoded.drop(columns=['cut', 'clarity'], errors='ignore')
#     clustering = Clustering(features)

#     elbow = clustering.elbow_with_silhouette(max_k=8)
#     clustering.elbow_curve(max_k=8, save_path='artifacts/clustering_elbow.png', show=False)
#     labels_kmeans = clustering.fit('kmeans', n_clusters=3)
#     labels_agglomerative = clustering.fit('agglomerative', n_clusters=3)
#     labels_meanshift = clustering.fit('mean_shift')

#     return {
#         'kmeans_inertia': clustering.get_kmeans_inertia(),
#         'elbow': elbow,
#         'kmeans_labels_count': len(labels_kmeans),
#         'agglomerative_labels_count': len(labels_agglomerative),
#         'meanshift_labels_count': len(labels_meanshift),
#     }

if __name__ == '__main__':
    analyzed_data = run_analyzer('diamonds.csv', 'cleaned_diamonds.csv')
    original_data = prepare_clean_data(pd.read_csv('diamonds.csv'))

    print('Choose data to run models on:')
    print('1) Whole cleaned data')
    print('2) Custom record count')
    data_choice = prompt_text('Enter 1 or 2 [default: 1]: ', default='1')

    if data_choice == '2':
        total_rows = len(original_data)
        while True:
            count_text = prompt_text(
                f'Enter number of records to use (1 to {total_rows}) [default: {total_rows}]: ',
                default=str(total_rows),
            )
            if count_text.isdigit() and int(count_text) > 0:
                requested_count = int(count_text)
                break
            print('Invalid input. Please enter a positive integer.')

        custom_count = min(requested_count, total_rows)
        selected_data = original_data.sample(n=custom_count, random_state=42)
        print(f'Using custom sample with {len(selected_data)} rows.')
    else:
        selected_data = original_data
        print(f'Using whole cleaned data with {len(selected_data)} rows.')

    print('Optional ANN hyperparameters for classification:')
    ann_config = {
        'epochs': prompt_positive_int('ANN epochs [default: 20]: ', 20),
        'batch_size': prompt_positive_int('ANN batch size [default: 32]: ', 32),
        'hidden_1': prompt_positive_int('ANN hidden layer 1 units [default: 64]: ', 64),
        'hidden_2': prompt_positive_int('ANN hidden layer 2 units [default: 32]: ', 32),
    }

    classification_metrics = run_classification(selected_data, ann_config=ann_config)
    plot_classification_metrics(classification_metrics, show=False)
    print('Classification metrics:', classification_metrics)

    svc_model_path, svc_holdout_accuracy = train_and_save_svc_model(selected_data)
    print(f'Saved SVC model: {svc_model_path}')
    print(f'SVC holdout accuracy before final refit: {svc_holdout_accuracy:.4f}')

    # regression_metrics = run_regression(selected_data)
    # plot_regression_metrics(regression_metrics, show=False)
    # print('Regression metrics:', regression_metrics)
    # clustering_metrics = run_clustering(selected_data)

    
   
    # print('Clustering metrics:', clustering_metrics)
