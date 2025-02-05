import numpy as np
import pickle as pkl
from common.common import shuffle

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.learning_rate = 0.0001
        self.loss_history = []
        self.loaded = False

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        n_samples, n_features = X.shape if X.ndim > 1 else (len(X), 1)

        # Initialize weights and bias based on y dimensions
        if y.ndim == 1:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
        else:  # Multi-output regression
            n_outputs = y.shape[1]
            self.weights = np.zeros((n_features, n_outputs))
            self.bias = np.zeros(n_outputs)

        # Split data into training (90%) and validation (10%)
        split_index = int(0.9 * n_samples)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        def mean_squared_error(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        def compute_loss(X_batch, y_batch):
            y_pred = np.dot(X_batch, self.weights) + self.bias
            mse = mean_squared_error(y_batch, y_pred)
            regularization_loss = self.regularization * np.sum(self.weights ** 2)
            return mse + regularization_loss

        # Declare variables used by the fit method
        best_loss = float('inf')
        best_weights, best_bias = None, None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            # Shuffle data at the start of each epoch
            X_train, y_train = shuffle(X_train, y_train)

            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                # Compute predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias
                error = y_pred - y_batch

                # Compute gradients
                grad_w = (2 / len(X_batch)) * np.dot(X_batch.T, error) + self.regularization * self.weights
                grad_b = (2 / len(X_batch)) * np.sum(error)

                # Update parameters
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Compute validation loss
            val_loss = compute_loss(X_val, y_val)
            self.loss_history.append(val_loss)

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias
                epochs_without_improvement = 0
            else:

                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    self.weights = best_weights
                    self.bias = best_bias
                    print(f"Early stopping at epoch {epoch} with best validation loss {best_loss:.6f}")
                    break

        # Restore best parameters
        self.weights = best_weights
        self.bias = best_bias

        print(f"Training completed with final validation loss {best_loss:.6f}")




    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        return np.zeros(X.shape[0], dtype=int)

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def save(self, filename):
        """Save the model to a file.

        Parameters
        ----------
        filename: str
            The name of the file.
        """
        with open(filename, 'wb') as f:
            # The below comment is to suppress the warning from PyCharm
            # noinspection PyTypeChecker
            pkl.dump(self, f)

    def load(self, filename):
        """Load the model from a file.

        Parameters
        ----------
        filename: str
            The name of the file.
        """
        try:
            with open(filename, 'rb') as f:
                model = pkl.load(f)
            self.weights = model.weights
            self.bias = model.bias
            self.loss_history = model.loss_history
            self.learning_rate = model.learning_rate
            self.batch_size = model.batch_size
            self.regularization = model.regularization
            self.max_epochs = model.max_epochs
            self.patience = model.patience
            self.learning_rate = model.learning_rate
            self.loaded = True
            print(f"Model loaded from {filename}")
            return self
        except FileNotFoundError:
            print(f"File {filename} not found")
            return None