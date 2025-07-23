import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our implementations
from classical_ml.logistic_regression import LogisticRegression
from classical_ml.svm import SupportVectorMachine
from classical_ml.naive_bayes import NaiveBayes
from classical_ml.random_forest import RandomForest
from classical_ml.pca import PCA
from utils.metrics import accuracy_score
from utils.visualization import plot_decision_boundary

def main():
    # Generate synthetic dataset
    X, y = make_moons(n_samples=100, noise=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train models
    models = {
        'Logistic Regression': LogisticRegression(learning_rate=0.01, n_iterations=1000),
        'SVM': SupportVectorMachine(learning_rate=0.001, n_iterations=1000),
        'Naive Bayes': NaiveBayes(),
        'Random Forest': RandomForest(n_trees=10, max_depth=5)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy:.4f}')
        
        # Plot decision boundary
        plot_decision_boundary(X_test_scaled, y_test, model, 
                             title=f'{name} Decision Boundary')
    
    # Demonstrate PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    print(f'\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}')

if __name__ == '__main__':
    main()
