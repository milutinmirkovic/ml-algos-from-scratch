import pandas as pd
import numpy as np


class KMeans:

    def __init__(self,
                k,
                max_iterations = 100,
                n_retries = 1,
                weights:np.array = None,
                initialization_method:str = "random"):

        assert initialization_method in {"random","kmeans++"} , "Invalid initialization method"

        self.k = k
        self.max_iterations = max_iterations
        self.n_retries = n_retries
        self.weights = weights
        self.initialization_method = initialization_method


    def _normalize(self,X: np.ndarray):

        self._mean_values = X.mean(axis=0)
        self._std_values = X.std(axis=0)

        return (X - self._mean_values) / self._std_values

    def _initialize_centroids(self,X):

        if self.initialization_method == "kmeans++":
            return self._initialize_centroids_pp(X)
        
        return self._initialize_centroids_random(X)
        

        
        
    def _initialize_centroids_random(self,X:np.ndarray):

        indices = np.random.choice(X.shape[0],self.k,replace=False)
        return X[indices]


    def _initialize_centroids_pp(self,X:np.ndarray):
        
        centroids = []
        centroids.append(X[np.random.randint(X.shape[0])])

        for i in range(self.k-1):
            dists = []
            for point in X:
                distance = min([np.sqrt(np.sum((point - c)**2)) for c in centroids]) 
                dists.append(distance)

            next_centroid = X[np.argmax(dists)]
            centroids.append(next_centroid)

        return np.array(centroids)


    def _assign_clusters(self,X,centroids):

        n_rows = X.shape[0]
        assigned = np.zeros(n_rows)

        for i in range(n_rows):

            point = X[i]
            distances = []
            for c in centroids:
                if self.weights is not None:
                    dist = np.sqrt(np.sum(((point - c) ** 2)*self.weights))
                else:
                    dist = (np.sum((point -c) ** 2)) ** (1/2)
                distances.append(dist)
                
            closest_centroid = np.argmin(distances)
            assigned[i] = closest_centroid

        return assigned

    def _update_centroids(self, X:np.ndarray, assignments: np.ndarray):

        centroids = np.zeros((self.k, X.shape[1]))

        for cluster in range(self.k):

            subset = X[assignments == cluster]
            centroids[cluster]  = subset.mean(axis=0)

        return centroids

    def learn(self, X: pd.DataFrame, normalize_features:bool = False):

        X = X.to_numpy()
        if normalize_features:
            X = self._normalize(X)

        best_centroids = None
        best_assignments = None
        best_quality = float('inf')

        if self.weights is not None and self.weights.shape != (X.shape[1],):
            raise ValueError(f"Tezine moraju da odgovaraju broju atribura skupa podataka!")
        
        for attempt in range(self.n_retries):

            centroids = self._initialize_centroids(X)
            old_quality = float('inf')

            for iteration in range(self.max_iterations):
                assignments = self._assign_clusters(X,centroids)
                new_centroids = self._update_centroids(X,assignments)

                centroids = new_centroids

                total_quality = self._calculate_quality(X, assignments)
                if total_quality == old_quality: 
                    break
                old_quality = total_quality

            if total_quality <best_quality:
                best_quality = total_quality
                best_centroids = centroids
                best_assignments = assignments

        self.centroids = best_centroids
        self.assignments = best_assignments
        self.quality = best_quality

        self.alerts =  self._diagnostics(X)

        return self





    def _calculate_quality(self,X:np.ndarray, assignments:np.ndarray):

        total_quality = 0

        for cluster_id in range(self.k):
            cluster_points = X[assignments==cluster_id]

            if len(cluster_points) == 0:
                continue

            cluster_var = np.var(cluster_points, axis=0)
            cluster_quality = cluster_var.sum() * len(cluster_points)
            total_quality += cluster_quality

        return total_quality

    def _diagnostics(self, X, spread_factor=2.0, min_separation=3.0):

        alerts = []

        for i in range(self.k):

            members = X[self.assignments == i]
            if len(members) > 0:

                dists = []
                for point in members:
                    dist = np.sqrt(np.sum((point - self.centroids[i]) ** 2))
                    dists.append(dist)

                avg_dist = np.mean(dists)
                max_dist = np.max(dists)

                if max_dist > spread_factor * avg_dist:
                    alerts.append(f"Cluster {i} is loose: max dist = {max_dist:.2f}")


            for j in range(i + 1, self.k):
                sep = np.sqrt(np.sum((self.centroids[i] - self.centroids[j]) ** 2))
                if sep < min_separation:
                    alerts.append(
                        f"Clusters {i} and {j} are too close: centroid dist = {sep:.2f}"
                    )

        return alerts
        