# Basic Machine Learning and Deep Learning Implementations

This repository contains implementations of various machine learning and deep learning algorithms from scratch using NumPy. These implementations are meant for educational purposes and to understand the underlying mechanics of these algorithms.

## Project Structure

```
basicML/
├── classical_ml/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   ├── knn.py
│   ├── svm.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── pca.py
│   └── dbscan.py
├── deep_learning/
│   ├── layers.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── neural_network.py
│   ├── conv_layers.py
│   ├── lstm.py
│   └── attention.py
├── utils/
│   ├── data_preprocessing.py
│   ├── metrics.py
│   └── visualization.py
└── examples/
    └── model_comparison.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SudarshanHegde2004/Basic-ML-DL.git
   cd Basic-ML-DL
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Algorithms Implemented

### Classical Machine Learning
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors
- K-Means Clustering
- Decision Trees
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- Principal Component Analysis (PCA)
- DBSCAN (Density-Based Spatial Clustering)

### Deep Learning
- Neural Network with customizable layers
- Convolutional Neural Network components
- LSTM (Long Short-Term Memory)
- Attention Mechanism
- Multi-Head Attention
- Various activation functions (ReLU, Sigmoid)
- Different optimizers (SGD, Adam)
- Loss functions (MSE, Binary Cross Entropy, Categorical Cross Entropy)

### Utilities
- Data preprocessing functions
- Metrics calculation
- Visualization tools
- Model comparison examples

## Usage

Each algorithm is implemented as a class with common methods:

- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions

Example:
```python
from classical_ml.logistic_regression import LogisticRegression

# Create and train model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

See `examples/model_comparison.py` for more detailed usage examples.

## Contributing

Feel free to contribute by:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
