# Basic Machine Learning and Deep Learning Implementations

This repository contains implementations of various machine learning and deep learning algorithms from scratch using NumPy. The implementations are meant for educational purposes and to understand the underlying mechanics of these algorithms.

## Project Structure

```
basicML/
├── classical_ml/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   └── knn.py
├── deep_learning/
│   ├── layers.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── neural_network.py
│   └── conv_layers.py
└── utils/
    ├── data_preprocessing.py
    ├── metrics.py
    └── visualization.py
```

## Installation

1. Clone the repository
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

### Deep Learning
- Neural Network with customizable layers
- Convolutional Neural Network components
- Various activation functions (ReLU, Sigmoid)
- Different optimizers (SGD, Adam)
- Loss functions (MSE, Binary Cross Entropy, Categorical Cross Entropy)

### Utilities
- Data preprocessing functions
- Metrics calculation
- Visualization tools

## Usage

Each algorithm is implemented as a class with common methods:

- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions

Example:
```python
from classical_ml.linear_regression import LinearRegression

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Contributing

Feel free to contribute by:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
