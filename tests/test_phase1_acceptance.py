import unittest
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from Analyzer import Analyzer
from Classifier import Classifier
from Regressor import Regressor
from Clustering import Clustering


class TestPhase1Acceptance(unittest.TestCase):
    def test_a1_analyzer_preprocess_load_drop_missing_drop_column(self):
        sample_path = Path("tests/data/sample_diamonds.csv")
        analyzer = Analyzer(str(sample_path))

        analyzer.preprocess_data()

        self.assertNotIn("Unnamed: 0", analyzer.data.columns)
        self.assertFalse(analyzer.data.isna().any().any())
        self.assertGreater(len(analyzer.data), 0)

    def test_a2_analyzer_sample_reduction_factor_and_validation(self):
        sample_path = Path("tests/data/sample_diamonds.csv")
        analyzer = Analyzer(str(sample_path))
        analyzer.drop_missing_data()
        cleaned_len = len(analyzer.data)

        sampled = analyzer.sample(0.4)
        self.assertEqual(len(sampled), int(round(cleaned_len * 0.4)))

        with self.assertRaises(ValueError):
            analyzer.sample(0.0)

        with self.assertRaises(ValueError):
            analyzer.sample(1.1)

    def test_a3_classifier_unified_fit_predict_score(self):
        X, y = make_classification(
            n_samples=120,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["cut"] = y

        classifier = Classifier(df, target="cut")
        X_train, X_test, y_train, y_test = train_test_split(
            classifier.X, classifier.y, test_size=0.2, random_state=42
        )

        classifier.fit("knn", X_train, y_train, k=3)
        preds_knn = classifier.predict("knn", X_test)
        self.assertEqual(len(preds_knn), len(y_test))
        acc_knn = classifier.score("knn", X_test, y_test, metric="accuracy")
        self.assertGreaterEqual(acc_knn, 0.0)
        self.assertLessEqual(acc_knn, 1.0)

        classifier.fit("decision_tree", X_train, y_train)
        acc_tree = classifier.score("decision_tree", X_test, y_test, metric="accuracy")
        self.assertGreaterEqual(acc_tree, 0.0)
        self.assertLessEqual(acc_tree, 1.0)

    def test_a4_regressor_unified_fit_predict_score(self):
        X, y = make_regression(
            n_samples=140,
            n_features=5,
            noise=5.0,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["price"] = y

        regressor = Regressor(df, target="price")
        X_train, X_test, y_train, y_test = train_test_split(
            regressor.X, regressor.y, test_size=0.2, random_state=42
        )

        regressor.fit("knn", X_train, y_train, k=3)
        preds_knn = regressor.predict("knn", X_test)
        self.assertEqual(len(preds_knn), len(y_test))
        scores_knn = regressor.score("knn", X_test, y_test)
        self.assertEqual(set(scores_knn.keys()), {"MSE", "RMSE", "MAE", "R2"})

        regressor.fit("decision_tree", X_train, y_train)
        scores_tree = regressor.score("decision_tree", X_test, y_test)
        self.assertEqual(set(scores_tree.keys()), {"MSE", "RMSE", "MAE", "R2"})

    def test_a5_clustering_kmeans_fit_labels_inertia(self):
        X, _ = make_classification(
            n_samples=90,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        clustering = Clustering(df)
        labels = clustering.fit("kmeans", n_clusters=3)

        self.assertEqual(len(labels), len(df))
        inertia = clustering.get_kmeans_inertia()
        self.assertIsInstance(inertia, float)


if __name__ == "__main__":
    unittest.main()
