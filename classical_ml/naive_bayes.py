import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def _calculate_likelihood(self, x, mean, var):
        """Gaussian likelihood of the data"""
        eps = 1e-4  # For numerical stability
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / (var + eps))
        return coeff * exponent
    
    def predict(self, X):
        y_pred = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._calculate_likelihood(x, self.mean[idx, :], self.var[idx, :])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Return class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)
