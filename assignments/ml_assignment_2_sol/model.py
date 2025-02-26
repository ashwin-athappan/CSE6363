from typing import Any

import numpy as np
from numpy import floating

from NeuralNetwork import Linear


def mean_squared_log_error(predictions: np.ndarray, targets: np.ndarray) -> floating[Any]:
    epsilon = 1e-9  # Small positive constant
    targets_array = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    predictions = np.clip(predictions, epsilon, None)
    targets_array = np.clip(targets_array, epsilon, None)
    return np.mean((np.log(predictions) - np.log(targets_array)) ** 2)


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> floating[Any]:
    targets_array = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    return np.mean((targets_array - predictions) ** 2)


def compute_gradient(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    targets_array = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    return 2 * (predictions - targets_array) / len(targets_array)


def train(m, X_train, y_train, X_val, y_val, learning_rate=0.1, epochs=50, patience=3):
    training_losses = []
    val_losses = []
    train_msle_losses = []
    val_msle_losses = []

    best_val_loss = float('inf')
    patience_counter = 0  # Tracks epochs without improvement

    for epoch in range(epochs):
        # Forward pass and loss computation
        predictions = m.forward(X_train)
        loss = mean_squared_error(predictions, y_train)
        msle_loss = mean_squared_log_error(predictions, y_train)
        train_msle_losses.append(msle_loss)
        training_losses.append(loss)

        # Compute gradient and back propagate
        gradient = compute_gradient(predictions, y_train)
        m.backward(gradient)

        # Update weights and biases
        for layer in m.layers:
            if isinstance(layer, Linear):
                layer.weights -= learning_rate * layer.weights
                layer.bias -= learning_rate * layer.bias

        # Compute validation loss
        val_predictions = m.forward(X_val)
        val_loss = mean_squared_error(val_predictions, y_val)
        val_msle_loss = mean_squared_log_error(val_predictions, y_val)
        val_msle_losses.append(val_msle_loss)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (No improvement in {patience} consecutive epochs)")
            break  # Stop training

        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

    losses = {
        'training': training_losses,
        'validation': val_losses,
        'train_msle': train_msle_losses,
        'val_msle': val_msle_losses
    }

    return losses