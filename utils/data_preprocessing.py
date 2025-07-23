import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Load and preprocess data with standardization
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def normalize_data(X):
    """
    Normalize data using MinMaxScaler
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to the dataset
    """
    poly_features = []
    for i in range(2, degree + 1):
        poly_features.append(np.power(X, i))
    return np.column_stack([X] + poly_features)
