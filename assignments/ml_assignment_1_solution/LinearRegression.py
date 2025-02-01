import numpy as np

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
        self.learning_rate = 0.01

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
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        n_samples, n_features = 0, 0
        try:
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
        except ValueError:
            n_samples = len(X)
            n_features = 1
            self.weights = 0

        self.bias = 0

        # Split data into training and validation sets (90% train, 10% validation)
        split_index = int(0.9 * n_samples)
        X_train, X_validation = X[:split_index], X[split_index:]
        y_train, y_validation = y[:split_index], y[split_index:]

        def mean_squared_error(y_true, y_predicted):
            """Compute the mean squared error.
            sum((actual - predicted) ^ 2) / n"""
            mse = np.sum((y_true - y_predicted)**2) / len(y_true)
            return mse

        def compute_loss(X_value, y_value, weights, bias):
            y_predicted = np.dot(X_value, weights) + bias
            mse = mean_squared_error(y_value, y_predicted)
            l2_term = regularization * np.sum(weights**2)
            return mse + l2_term


        def shuffle(X, y):
            """Shuffle the data."""
            indices = np.random.permutation(len(X))
            return X[indices], y[indices]

        best_loss = float('inf')
        best_weights = None
        best_bias = None
        epochs_without_improvement = 0

        # TODO: Implement the training loop.
        for epoch in range(max_epochs):
            # X_train, y_train = shuffle(X_train, y_train)

            # Perform batch gradient descent
            for i in range(0, len(X_train), batch_size):
                X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]

                # Compute the predictions
                predicted_value = np.dot(X_batch, self.weights) + self.bias

                # Compute the gradients
                error = predicted_value - y_batch
                gradient_weights = (2 / len(X_batch)) * np.dot(X_batch.T, error) + 2 * regularization * self.weights
                gradient_bias = (2 / len(X_batch)) * np.sum(error)

                # Update the weights and bias
                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

            # Compute the validation loss
            val_loss = compute_loss(X_validation, y_validation, self.weights, self.bias)

            # Stop the training if the validation loss does not improve for `patience` epochs
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        self.weights = best_weights
        self.bias = best_bias
        print(f"Training completed after {max_epochs} epochs with validation loss {best_loss}")




    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        pass