from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.inertias_ = []

    def fit(self, model_name, X=None, **params):
        model_name = model_name.lower()
        if X is None:
            X = self.data

        if model_name == 'kmeans':
            model = KMeans(
                n_clusters=params.get('n_clusters', 3),
                random_state=params.get('random_state', 42),
            )
            labels = model.fit_predict(X)
            self.models['kmeans'] = model
            return labels

        if model_name == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 2))
            labels = model.fit_predict(X)
            self.models['agglomerative'] = model
            return labels

        if model_name == 'mean_shift':
            model = MeanShift()
            labels = model.fit_predict(X)
            self.models['mean_shift'] = model
            return labels

        raise ValueError("Unsupported model. Use one of: kmeans, agglomerative, mean_shift")

    def get_kmeans_inertia(self):
        model = self.models.get('kmeans')
        if model is None:
            raise ValueError("KMeans model has not been trained yet.")
        return model.inertia_

    def elbow_with_silhouette(self, max_k=10):
        inertias = []
        silhouettes = {}
        for k in range(1, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(self.data)
            inertias.append(model.inertia_)
            if k > 1:
                silhouettes[k] = silhouette_score(self.data, labels)
        self.inertias_ = inertias
        return {'inertias': inertias, 'silhouette_scores': silhouettes}

    def elbow_curve(self, max_k=10, save_path=None, show=True):
        inertias = []
        for k in range(1, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(self.data)
            inertias.append(model.inertia_)

        self.inertias_ = inertias
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
        return inertias

    def kmeans_clustering(self, max_k=10):
        self.inertias_ = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            self.inertias_.append(kmeans.inertia_)
            if k > 1:
                score = silhouette_score(self.data, kmeans.labels_)
                print(f'K={k}, Silhouette Score: {score}')
        plt.plot(range(1, max_k + 1), self.inertias_, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.tight_layout()
        plt.show()

    def agglomerative_clustering(self, n_clusters=2):
        labels = self.fit('agglomerative', self.data, n_clusters=n_clusters)
        self.models['Agglomerative'] = self.models['agglomerative']
        print(f'Agglomerative Clustering with {n_clusters} clusters completed.')
        return labels

    def mean_shift_clustering(self):
        labels = self.fit('mean_shift', self.data)
        self.models['MeanShift'] = self.models['mean_shift']
        print('Mean-Shift Clustering completed.')
        return labels

    def predict(self, model_name, new_data):
        key = model_name.lower()
        model = self.models.get(key) or self.models.get(model_name)
        if model is None:
            raise ValueError(f'Model {model_name} not found. Train the model first.')
        if not hasattr(model, 'predict'):
            raise ValueError(f'The model {model_name} does not support prediction.')
        return model.predict(new_data)
