import numpy as np
import pandas as pd
import pickle


class Layer:
    def forward(self, inp):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.inp = None
        # Xavier/Glorot initialization with smaller scale
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / (input_dim + output_dim))
        self.bias = np.zeros((1, output_dim))

    def forward(self, inp):
        self.inp = inp
        return np.dot(self.inp, self.weights) + self.bias

    def backward(self, grad_output, learning_rate=0.1):
        # Clip gradients to prevent explosion
        grad_output = np.clip(grad_output, -1e3, 1e3)

        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inp.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Clip gradients for weight updates
        grad_weights = np.clip(grad_weights, -1e3, 1e3)
        grad_bias = np.clip(grad_bias, -1e3, 1e3)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inp):
        # Clip input to prevent overflow
        inp = np.clip(inp, -500, 500)
        self.output = 1 / (1 + np.exp(-inp))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class ReLU(Layer):
    def __init__(self):
        self.inp = None

    def forward(self, inp):
        self.inp = inp
        return np.maximum(0, inp)

    def backward(self, grad_output):
        # Use where to avoid invalid value in multiply
        grad_input = np.where(self.inp > 0, grad_output, 0)
        return grad_input


class MSELoss:
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets.reshape(predictions.shape)
        # Use np.square for better numerical stability
        diff = self.predictions - self.targets
        return np.mean(np.square(diff))

    def backward(self):
        n_samples = self.targets.shape[0]
        # Clip the gradient to prevent explosion
        grad = np.clip(2 * (self.predictions - self.targets) / n_samples, -1e3, 1e3)
        return grad


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
            # Add gradient checking
            if np.any(np.isnan(inp)) or np.any(np.isinf(inp)):
                print(f"Warning: NaN or Inf detected in forward pass at {layer.__class__.__name__}")
                inp = np.nan_to_num(inp, nan=0.0, posinf=1e3, neginf=-1e3)
        return inp

    def backward(self, grad_output, learning_rate=0.1):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                grad_output = layer.backward(grad_output, learning_rate)
            else:
                grad_output = layer.backward(grad_output)
            # Add gradient checking
            if np.any(np.isnan(grad_output)) or np.any(np.isinf(grad_output)):
                print(f"Warning: NaN or Inf detected in backward pass at {layer.__class__.__name__}")
                grad_output = np.nan_to_num(grad_output, nan=0.0, posinf=1e3, neginf=-1e3)

    def save_weights(self, filename):
        weights = [(layer.weights, layer.bias) for layer in self.layers if isinstance(layer, Linear)]
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights, layer.bias = weights[idx]
                idx += 1


def preprocess_data(X, y=None, is_training=True):
    df_X = pd.DataFrame(X)

    # Convert to numeric, handling errors
    df_X = df_X.apply(pd.to_numeric, errors='coerce')

    # Drop columns with too many missing values
    if is_training:
        cols_to_keep = df_X.columns[df_X.isnull().mean() < 0.3]
        df_X = df_X[cols_to_keep]

    # Fill remaining missing values
    df_X = df_X.fillna(df_X.mean())

    # Robust scaling to handle outliers
    if is_training:
        q1 = df_X.quantile(0.25)
        q3 = df_X.quantile(0.75)
        iqr = q3 - q1
        df_X = (df_X - q1) / iqr

    # Clip extreme values
    df_X = df_X.clip(-5, 5)

    X_processed = df_X.to_numpy(dtype=np.float32)

    if y is not None:
        # Log transform for target variable (if all values are positive)
        if np.all(y > 0):
            y_processed = np.log1p(y)
        else:
            y_processed = y
        y_processed = np.array(y_processed, dtype=np.float32).reshape(-1, 1)
        return X_processed, y_processed

    return X_processed


def train_model(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    n_samples = X_train.shape[0]
    loss_fn = MSELoss()
    losses = []

    # Learning rate decay
    initial_lr = learning_rate

    for epoch in range(epochs):
        # Decay learning rate
        current_lr = initial_lr / (1 + epoch * 0.01)

        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Forward pass
            output = model.forward(X_batch)
            loss = loss_fn.forward(output, y_batch)

            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: Invalid loss value at epoch {epoch}, batch {i // batch_size}")
                continue

            epoch_losses.append(loss)

            # Backward pass
            grad_loss = loss_fn.backward()
            model.backward(grad_loss, current_lr)

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")

    return losses


# Example usage
if __name__ == "__main__":
    # Load and preprocess the data
    dataset = np.load('nyc_taxi_data.npy', allow_pickle=True).item()
    X_train, y_train = preprocess_data(dataset["X_train"], dataset["y_train"])

    # Create the model with smaller architecture
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Linear(input_dim, 32))
    model.add(ReLU())
    model.add(Linear(32, 16))
    model.add(ReLU())
    model.add(Linear(16, 1))

    # Train the model with adjusted parameters
    losses = train_model(model, X_train, y_train,
                         epochs=100,
                         batch_size=64,  # Increased batch size for stability
                         learning_rate=0.0001)  # Reduced learning rate

    # Save the trained model
    model.save_weights("XOR_solved.w")