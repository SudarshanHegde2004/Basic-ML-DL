import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        
    def _get_neighbors(self, X, point_idx):
        """Find all points within eps distance of point_idx"""
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)  # -1 represents unassigned points
        
        # Current cluster label
        cluster_label = 0
        
        # Iterate through each point
        for point_idx in range(n_samples):
            # Skip if point is already assigned
            if self.labels[point_idx] != -1:
                continue
                
            # Find neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            # Mark as noise if not enough neighbors
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -2  # -2 represents noise
                continue
                
            # Start a new cluster
            cluster_label += 1
            self.labels[point_idx] = cluster_label
            
            # Process neighbors
            seed_set = neighbors.tolist()
            for neighbor_idx in seed_set:
                # If point was noise, add it to cluster
                if self.labels[neighbor_idx] == -2:
                    self.labels[neighbor_idx] = cluster_label
                    
                # Skip if point is already assigned
                if self.labels[neighbor_idx] != -1:
                    continue
                    
                # Add point to cluster
                self.labels[neighbor_idx] = cluster_label
                
                # Find neighbors of neighbor
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # Add new core points to seed set
                if len(new_neighbors) >= self.min_samples:
                    seed_set.extend(new_neighbors)
        
        return self.labels
