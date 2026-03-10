import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class Classifier:
    def __init__(self, data, target):
        self.X = data.drop(columns=[target])
        self.y = data[target]

        # Ensure the target variable is treated as categorical
        if self.y.dtypes != 'int' and self.y.dtypes != 'category':
            self.y = LabelEncoder().fit_transform(self.y)

        self.models = {}
        self.supported_models = {
            'logistic_regression',
            'knn',
            'decision_tree',
            'random_forest',
            'svc',
            'ann',
        }

    def _build_model(self, model_name, **params):
        if model_name == 'logistic_regression':
            return LogisticRegression(
                max_iter=params.get('max_iter', 1000),
                random_state=params.get('random_state', 42),
            )
        if model_name == 'knn':
            return KNeighborsClassifier(n_neighbors=params.get('k', params.get('n_neighbors', 5)))
        if model_name == 'decision_tree':
            return DecisionTreeClassifier(
                criterion=params.get('criterion', 'gini'),
                random_state=params.get('random_state', 42),
            )
        if model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                criterion=params.get('criterion', 'gini'),
                random_state=params.get('random_state', 42),
            )
        if model_name == 'svc':
            return SVC(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                random_state=params.get('random_state', 42),
            )
        if model_name == 'ann':
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense
            except ImportError as exc:
                raise ImportError("TensorFlow/Keras is required for ANN model.") from exc

            input_dim = params['input_dim']
            n_classes = params.get('n_classes')
            if n_classes is None:
                n_classes = len(set(self.y))
            model = Sequential([
                Dense(params.get('hidden_1', 64), activation=params.get('activation_1', 'relu'), input_dim=input_dim),
                Dense(params.get('hidden_2', 32), activation=params.get('activation_2', 'relu')),
                Dense(n_classes, activation='softmax')
            ])
            model.compile(
                optimizer=params.get('optimizer', 'adam'),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            return model
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {sorted(self.supported_models)}")

    def fit(self, model_name, X_train, y_train, **params):
        model_name = model_name.lower()
        model = self._build_model(model_name, **params)
        if model_name == 'ann':
            model.fit(
                X_train,
                y_train,
                epochs=params.get('epochs', 10),
                batch_size=params.get('batch_size', 32),
                verbose=0,
            )
        else:
            model.fit(X_train, y_train)
        self.models[model_name] = model
        return model

    def tune_knn(self, k_values=None, test_size=0.2, random_state=42):
        if k_values is None:
            k_values = [3, 5, 7, 9]
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        best_k = None
        best_acc = -1
        scores = {}
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
             
            scores[k] = acc
            if acc > best_acc:
                best_acc = acc
                best_k = k
        return best_k, best_acc, scores

    def train_knn(self, k=3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('knn', X_train, y_train, k=k)
        y_pred = model.predict(X_test)
        self.models['KNN'] = model
        print(f'KNN Accuracy: {accuracy_score(y_test, y_pred)}')
        print(confusion_matrix(y_test, y_pred))

    def train_logistic_regression(self, max_iter=1000):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('logistic_regression', X_train, y_train, max_iter=max_iter)
        y_pred = model.predict(X_test)
        self.models['LogisticRegression'] = model
        print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}')
        print(confusion_matrix(y_test, y_pred))

    def train_decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('decision_tree', X_train, y_train)
        y_pred = model.predict(X_test)
        self.models['DecisionTree'] = model
        print(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred)}')
        print(confusion_matrix(y_test, y_pred))

    def predict(self, model_name, X):
        key = model_name.lower()
        model = self.models.get(key) or self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        if key == 'ann':
            return model.predict(X).argmax(axis=1)
        return model.predict(X)

    def score(self, model_name, X, y, metric='accuracy'):
        y_pred = self.predict(model_name, X)
        if callable(metric):
            return metric(y, y_pred)
        metric_name = metric.lower()
        if metric_name in ('accuracy', 'accuracy_score'):
            return accuracy_score(y, y_pred)
        if metric_name == 'f1_macro':
            return f1_score(y, y_pred, average='macro')
        if metric_name == 'precision_macro':
            return precision_score(y, y_pred, average='macro', zero_division=0)
        if metric_name == 'recall_macro':
            return recall_score(y, y_pred, average='macro', zero_division=0)
        raise ValueError("Unsupported metric. Use one of: accuracy, f1_macro, precision_macro, recall_macro")

    def train_random_forest(self, n_estimators=100, criterion='gini'):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('random_forest', X_train, y_train, n_estimators=n_estimators, criterion=criterion)
        y_pred = model.predict(X_test)
        self.models['RandomForest'] = model
        print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred)}')
        print(confusion_matrix(y_test, y_pred))

    def train_svc(self, kernel='rbf', C=1.0):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('svc', X_train, y_train, kernel=kernel, C=C)
        y_pred = model.predict(X_test)
        self.models['SVC'] = model
        print(f'SVC Accuracy: {accuracy_score(y_test, y_pred)}')
        print(confusion_matrix(y_test, y_pred))

    def train_ann(self, input_dim, epochs=10, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('ann', X_train, y_train, input_dim=input_dim, epochs=epochs, batch_size=batch_size)
        y_pred = self.predict('ann', X_test)
        self.models['ANN'] = model
        print(f'ANN Accuracy: {accuracy_score(y_test, y_pred)}')
        print(confusion_matrix(y_test, y_pred))

    def plot_confusion_matrix(self, model_name, X, y, save_path=None, show=True):
        y_pred = self.predict(model_name, X)
        cm = confusion_matrix(y, y_pred)
        ConfusionMatrixDisplay.from_predictions(y, y_pred)
        if save_path:
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
        return cm
