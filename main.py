from Analyzer import Analyzer
from Classifier import Classifier
from Regressor import Regressor
from Clustering import Clustering
import pandas as pd
from sklearn.model_selection import train_test_split


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
    encoded = df.copy()
    categorical_columns = encoded.select_dtypes(include=['object', 'string', 'category']).columns
    for col in categorical_columns:
        encoded[col] = encoded[col].astype('category').cat.codes
    return encoded


def run_analyzer(data_path='diamonds.csv', output_path='cleaned_diamonds.csv'):
    analyzer = Analyzer(data_path)
    analyzer.show_info()
    analyzer.preprocess_data()
    # Enforce shared cleaning policy before artifacts and export.
    analyzer.data = prepare_clean_data(analyzer.data)
    analyzer.save_cleaned_data(output_path)
    analyzer.plot_correlation_matrix(save_path='artifacts/correlation_matrix.png', show=False)
    analyzer.plot_histograms_numerical(save_path='artifacts/histograms_numerical.png', show=False)
    analyzer.plot_histograms_categorical(save_dir='artifacts/categorical_histograms', show=False)
    analyzer.plot_pairPlot(save_path='artifacts/pairplot.png', show=False)
    analyzer.plot_boxPlot('price', save_path='artifacts/boxplot_price.png', show=False)
    return analyzer.data


def run_classification(data):
    cleaned = prepare_clean_data(data)
    features = cleaned.drop(columns=['cut'])
    # One-hot encoding prevents introducing false order between category values.
    encoded_features = pd.get_dummies(features, drop_first=False)
    classification_data = pd.concat([encoded_features, cleaned['cut'].reset_index(drop=True)], axis=1)

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

    metrics = {}
    for model in ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svc']:
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


def run_clustering(data):
    cleaned = prepare_clean_data(data)
    encoded = encode_categoricals(cleaned)
    features = encoded.drop(columns=['cut', 'clarity'], errors='ignore')
    clustering = Clustering(features)

    elbow = clustering.elbow_with_silhouette(max_k=8)
    clustering.elbow_curve(max_k=8, save_path='artifacts/clustering_elbow.png', show=False)
    labels_kmeans = clustering.fit('kmeans', n_clusters=3)
    labels_agglomerative = clustering.fit('agglomerative', n_clusters=3)
    labels_meanshift = clustering.fit('mean_shift')

    return {
        'kmeans_inertia': clustering.get_kmeans_inertia(),
        'elbow': elbow,
        'kmeans_labels_count': len(labels_kmeans),
        'agglomerative_labels_count': len(labels_agglomerative),
        'meanshift_labels_count': len(labels_meanshift),
    }

if __name__ == '__main__':
    analyzed_data = run_analyzer('diamonds.csv', 'cleaned_diamonds.csv')

    print('Choose data to run models on:')
    print('1) Whole cleaned data')
    print('2) Custom record count')
    data_choice = prompt_text('Enter 1 or 2 [default: 1]: ', default='1')

    if data_choice == '2':
        total_rows = len(analyzed_data)
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
        selected_data = analyzed_data.sample(n=custom_count, random_state=42)
        print(f'Using custom sample with {len(selected_data)} rows.')
    else:
        selected_data = analyzed_data
        print(f'Using whole cleaned data with {len(selected_data)} rows.')

    classification_metrics = run_classification(selected_data)
    regression_metrics = run_regression(selected_data)
    clustering_metrics = run_clustering(selected_data)

    print('Classification metrics:', classification_metrics)
    print('Regression metrics:', regression_metrics)
    print('Clustering metrics:', clustering_metrics)
