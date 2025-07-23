import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        
    def fit(self, X):
        # Initialize random centroids
        idx = np.random.permutation(X.shape[0])
        self.centroids = X[idx[:self.n_clusters]]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign samples to closest centroids
            distances = self._calculate_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    self.centroids[k] = np.mean(X[labels == k], axis=0)
            
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
                
    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k])**2, axis=1)
        return distances
