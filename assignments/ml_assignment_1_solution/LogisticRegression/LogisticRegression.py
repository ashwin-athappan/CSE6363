import numpy as np
import pickle as pkl
from common.common import shuffle

class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=1000, patience=3):
        """Logistic Regression using Gradient Descent.

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
        self.learning_rate = 0.1
        self.loss_history = []

    def softmax(self, z):
        # Adjusted softmax for multi-class classification
        exp_z = np.exp(z)  # Prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=1000, patience=3):
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
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        # One-hot encoding of labels
        # y_one_hot will contain 1 in the column of the class and 0 in the other columns
        y_one_hot = np.eye(num_classes)[y]

        # Initialize weights and bias based on y dimensions
        self.weights = np.zeros((n_features, num_classes))  # Adjust for multi-class
        self.bias = np.zeros(num_classes)  # Bias for each class

        best_loss = float('inf')
        best_weights, best_bias = None, None
        epochs_without_improvement = 0

        # Training Loop
        for epoch in range(self.max_epochs):

            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y_one_hot[i:i+self.batch_size]

                logits = np.dot(X_batch, self.weights) + self.bias
                probs = self.softmax(logits)  # Apply softmax to logits for probabilities

                # Compute gradients
                grad_w = np.dot(X_batch.T, (probs - y_batch)) / len(X_batch) + (self.regularization / len(X_batch)) * self.weights
                grad_b = np.mean(probs - y_batch, axis=0)

                # Update weights and bias
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Compute validation loss (cross-entropy loss with L2 regularization)
            val_logits = np.dot(X, self.weights) + self.bias
            val_probs = self.softmax(val_logits)
            val_loss = -np.mean(np.sum(y_one_hot * np.log(val_probs + 1e-9), axis=1))
            val_loss += (self.regularization / (2 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights
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
        """Predict using the logistic model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # Compute the logits (raw predictions)
        logits = np.dot(X, self.weights) + self.bias

        # Apply softmax to get probabilities for each class
        probs = self.softmax(logits)

        # Predict the class with the highest probability
        return np.argmax(probs, axis=1)  # Returns the index of the class with the highest probability

    def score(self, X, y):
        """Evaluate the logistic model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
