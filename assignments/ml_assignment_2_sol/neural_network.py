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
        self.weights = np.random.randn(output_dim, input_dim) * 0.01
        self.bias = np.zeros(output_dim)

    def forward(self, inp):
        self.inp = inp
        return np.dot(inp, self.weights.T) + self.bias

    def backward(self, grad_output, learning_rate=0.1):
        grad_input = np.dot(grad_output, self.weights)
        grad_weights = np.dot(grad_output.T, self.inp)
        grad_bias = np.sum(grad_output, axis=0)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inp):
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
        return grad_output * (self.inp > 0)


class BinaryCrossEntropyLoss:
    def __init__(self):
        self.targets = None
        self.predictions = None

    def forward(self, predictions, targets):
        self.predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        self.targets = targets
        return -np.mean(targets * np.log(self.predictions) + (1 - targets) * np.log(1 - self.predictions))

    def backward(self):
        return (self.predictions - self.targets) / (self.targets.shape[0])


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def backward(self, grad_output, learning_rate=0.1):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                grad_output = layer.backward(grad_output, learning_rate)
            else:
                grad_output = layer.backward(grad_output)

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


l = Linear(4, 5)
print(l.weights)

dataset = np.load('nyc_taxi_data.npy', allow_pickle=True).item()

# Convert X_train to a DataFrame
df_X_train = pd.DataFrame(dataset["X_train"])
df_X_train = df_X_train.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
df_X_train = df_X_train.dropna(axis=1)  # Drop non-numeric columns
X_train = df_X_train.to_numpy(dtype=np.float32)

# Ensure y_train is numeric
y_train = pd.to_numeric(dataset["y_train"], errors='coerce')
y_train = np.array(y_train, dtype=np.float32)


model = Sequential()
model.add(Linear(X_train.shape[1], 1))
model.add(Sigmoid())
model.add(Linear(X_train.shape[1], 1))
model.add(Sigmoid())

loss_fn = BinaryCrossEntropyLoss()

# Training loop
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    output = model.forward(X_train)
    loss = loss_fn.forward(output, y_train)
    grad_loss = loss_fn.backward()
    model.backward(grad_loss, learning_rate)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save trained weights
model.save_weights("XOR_solved.w")

