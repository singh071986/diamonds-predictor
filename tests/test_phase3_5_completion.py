import unittest

import pandas as pd
from sklearn.datasets import make_blobs

from main import run_classification, run_regression
from Clustering import Clustering


class TestPhase3To5Completion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('diamonds.csv').sample(n=4000, random_state=42).reset_index(drop=True)

    def test_phase3_classification_target_accuracy(self):
        metrics = run_classification(self.df)
        best_accuracy = max(metrics.values())
        self.assertGreaterEqual(best_accuracy, 0.50)

    def test_phase4_regression_target_r2(self):
        metrics = run_regression(self.df)
        best_r2 = max(model_metrics['R2'] for model_metrics in metrics.values())
        self.assertGreaterEqual(best_r2, 0.90)

    def test_phase5_clustering_diagnostics(self):
        X, _ = make_blobs(n_samples=200, centers=4, n_features=4, random_state=42)
        feature_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        clustering = Clustering(feature_df)

        result = clustering.elbow_with_silhouette(max_k=6)
        self.assertEqual(len(result['inertias']), 6)
        self.assertIn(2, result['silhouette_scores'])

        labels = clustering.fit('kmeans', n_clusters=4)
        self.assertEqual(len(labels), len(feature_df))
        self.assertIsInstance(clustering.get_kmeans_inertia(), float)


if __name__ == '__main__':
    unittest.main()
