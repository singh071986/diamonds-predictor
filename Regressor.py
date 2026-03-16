import importlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np


class Regressor:
    def __init__(self, data, target):
        self.X = data.drop(columns=[target])
        self.y = data[target]
        self.models = {}
        self.supported_models = {
            'linear_regression',
            'knn',
            'decision_tree',
            'random_forest',
            'svr',
            'ann',
        }

    def _build_model(self, model_name, **params):
        if model_name == 'linear_regression':
            return LinearRegression()
        if model_name == 'knn':
            return KNeighborsRegressor(n_neighbors=params.get('k', params.get('n_neighbors', 5)))
        if model_name == 'decision_tree':
            return DecisionTreeRegressor(
                criterion=params.get('criterion', 'squared_error'),
                random_state=params.get('random_state', 42),
            )
        if model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                criterion=params.get('criterion', 'squared_error'),
                random_state=params.get('random_state', 42),
            )
        if model_name == 'svr':
            return SVR(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
            )
        if model_name == 'ann':
            try:
                keras_models = importlib.import_module('tensorflow.keras.models')
                keras_layers = importlib.import_module('tensorflow.keras.layers')
                keras_optimizers = importlib.import_module('tensorflow.keras.optimizers')
            except ImportError as exc:
                raise ImportError("TensorFlow/Keras is required for ANN model.") from exc

            Sequential = keras_models.Sequential
            Dense = keras_layers.Dense
            Dropout = keras_layers.Dropout
            Adam = keras_optimizers.Adam

            input_dim = params['input_dim']
            layers = params.get('layers', [64, 32])
            dropout = params.get('dropout', 0.2)
            learning_rate = params.get('learning_rate', 0.001)
            model = Sequential()
            model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
            for units in layers[1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(dropout))
            model.add(Dense(1))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            return model
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {sorted(self.supported_models)}")

    def fit(self, model_name, X_train, y_train, **params):
        model_name = model_name.lower()
        model = self._build_model(model_name, **params)
        if model_name == 'ann':
            model.fit(
                X_train,
                y_train,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32),
                verbose=0,
                validation_data=params.get('validation_data'),
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
        best_r2 = float('-inf')
        scores = {}
        for k in k_values:
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            scores[k] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_k = k
        return best_k, best_r2, scores

    def tune_decision_tree(self, criteria=None, test_size=0.2, random_state=42):
        if criteria is None:
            criteria = ['squared_error', 'friedman_mse', 'absolute_error']
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        best_criterion = None
        best_r2 = float('-inf')
        scores = {}
        for criterion in criteria:
            model = DecisionTreeRegressor(criterion=criterion, random_state=random_state)
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            scores[criterion] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_criterion = criterion
        return best_criterion, best_r2, scores

    def tune_random_forest(self, n_estimators_values=None, criteria=None, test_size=0.2, random_state=42):
        if n_estimators_values is None:
            n_estimators_values = [100, 200]
        if criteria is None:
            criteria = ['squared_error', 'absolute_error']

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        best_params = None
        best_r2 = float('-inf')
        scores = {}
        for n_estimators in n_estimators_values:
            for criterion in criteria:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    random_state=random_state,
                )
                model.fit(X_train, y_train)
                r2 = r2_score(y_test, model.predict(X_test))
                key = f'n_estimators={n_estimators},criterion={criterion}'
                scores[key] = r2
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {'n_estimators': n_estimators, 'criterion': criterion}
        return best_params, best_r2, scores

    def train_knn(self, k=3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('knn', X_train, y_train, k=k)
        y_pred = model.predict(X_test)
        self.models['KNN'] = model
        print(f'KNN R2 Score: {r2_score(y_test, y_pred)}')
        print(f'KNN MSE: {mean_squared_error(y_test, y_pred)}')

    def train_linear_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('linear_regression', X_train, y_train)
        y_pred = model.predict(X_test)
        self.models['LinearRegression'] = model
        print(f'Linear Regression R2 Score: {r2_score(y_test, y_pred)}')
        print(f'Linear Regression MSE: {mean_squared_error(y_test, y_pred)}')

    def train_decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('decision_tree', X_train, y_train)
        y_pred = model.predict(X_test)
        self.models['DecisionTree'] = model
        print(f'Decision Tree R2 Score: {r2_score(y_test, y_pred)}')
        print(f'Decision Tree MSE: {mean_squared_error(y_test, y_pred)}')

    def train_random_forest(self, n_estimators=100, criterion='squared_error'):
        """
        Train a Random Forest Regressor.
        :param n_estimators: Number of trees in the forest.
        :param criterion: Function to measure the quality of a split ('mse' or 'mae').
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('random_forest', X_train, y_train, n_estimators=n_estimators, criterion=criterion)
        y_pred = model.predict(X_test)
        self.models['RandomForest'] = model
        print(f'Random Forest R2 Score: {r2_score(y_test, y_pred)}')
        print(f'Random Forest MSE: {mean_squared_error(y_test, y_pred)}')

    def train_svc(self, kernel='rbf', C=1.0):
        """
        Train a Support Vector Regressor (SVR).
        :param kernel: Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid').
        :param C: Regularization parameter.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit('svr', X_train, y_train, kernel=kernel, C=C)
        y_pred = model.predict(X_test)
        self.models['SVC'] = model
        print(f'SVC R2 Score: {r2_score(y_test, y_pred)}')
        print(f'SVC MSE: {mean_squared_error(y_test, y_pred)}')

    def train_ann(self, input_dim, layers=None, dropout=0.2, learning_rate=0.001, epochs=50, batch_size=32):
        """
        Train an Artificial Neural Network (ANN) Regressor.
        :param input_dim: Number of input features.
        :param layers: List defining the number of neurons in each hidden layer.
        :param dropout: Dropout rate for regularization.
        :param learning_rate: Learning rate for the optimizer.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """
        if layers is None:
            layers = [64, 32]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = self.fit(
            'ann',
            X_train,
            y_train,
            input_dim=input_dim,
            layers=layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
        )
        y_pred = model.predict(X_test).flatten()
        self.models['ANN'] = model
        print(f'ANN R2 Score: {r2_score(y_test, y_pred)}')
        print(f'ANN MSE: {mean_squared_error(y_test, y_pred)}')

    def predict(self, model_name, X_new):
        """
        Predict new data using a trained model.
        :param model_name: Name of the trained model (e.g., 'KNN', 'DecisionTree').
        :param X_new: New data to predict.
        :return: Predicted values.
        """
        key = model_name.lower()
        model = self.models.get(key) or self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} is not trained yet.")
        prediction = model.predict(X_new)
        if hasattr(prediction, 'flatten'):
            return prediction.flatten()
        return prediction

    

    def score(self, model_name, X_test, y_test):
        """
        Evaluate a trained model using regression metrics.
        :param model_name: Name of the trained model (e.g., 'KNN', 'DecisionTree').
        :param X_test: Test features.
        :param y_test: Test target values.
        :return: Dictionary of regression metrics (MSE, RMSE, MAE, R2).
        """
        y_pred = self.predict(model_name, X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
        }
