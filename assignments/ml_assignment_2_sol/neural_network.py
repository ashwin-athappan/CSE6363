import numpy as np
import json

class Layer:
    def forward(self, inp):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.inp = None
        self.weights = np.random.randn(input_dimension, output_dimension)
        self.bias = np.zeros((1, output_dimension))

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
        super().__init__()
        self.inp = None
        self.output = None

    def forward(self, inp):
        # Clip input to prevent overflow
        self.inp = inp
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


class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        self.predicted_probabilities = None
        self.true_labels = None

    def forward_with_labels(self, predicted_probabilities, true_labels):
        self.true_labels = true_labels
        return self.forward(predicted_probabilities)

    def forward(self, predicted_probabilities):
        # Clip predicted probabilities to avoid log(0) or log(1)
        self.predicted_probabilities = np.clip(predicted_probabilities, 1e-15, 1 - 1e-15)

        # Calculate binary cross-entropy loss
        loss = -np.mean(self.true_labels * np.log(self.predicted_probabilities) +
                        (1 - self.true_labels) * np.log(1 - self.predicted_probabilities))
        return loss

    def backward_with_labels(self, predicted_probabilities, true_labels):
        self.true_labels = true_labels
        return self.backward(predicted_probabilities)

    def backward(self, predicted_probabilities):
        # Compute the gradient of the loss with respect to the input
        input_gradient = (predicted_probabilities - self.true_labels) / (
                self.predicted_probabilities * (1 - self.predicted_probabilities) * self.true_labels.size)
        return input_gradient


class Sequential(Layer):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)


def save_weights(model, file_path):
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            weights[f'layer_{i}'] = layer.weights.tolist()
        if hasattr(layer, 'bias'):
            weights[f'layer_{i}_bias'] = layer.bias.tolist()
    with open(file_path, 'w') as f:
        json.dump(weights, f)
    print(f"Weights saved to {file_path}")


def load_weights(model, file_path):
    with open(file_path, 'r') as f:
        weights = json.load(f)
    for i, layer in enumerate(model.layers):
        if f'layer_{i}' in weights:
            layer.weights = np.array(weights[f'layer_{i}'])
        if f'layer_{i}_bias' in weights:
            layer.bias = np.array(weights[f'layer_{i}_bias'])
    print(f"Weights loaded from {file_path}")
