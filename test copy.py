import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize data as original input ranges from 0 to 255
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape data to be (num_samples, 784)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    # One hot encode the output
    num_classes = 10  # 10 classes
    y_train = to_categorical(y_train, num_classes).T
    y_test = to_categorical(y_test, num_classes).T

    return X_train, y_train, X_test, y_test

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01  # Transposed compared to your original
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def relu_derivatives(Z):
    return (Z > 0).astype(int)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_crossentropy_loss(Y, Y_hat):
    Y_hat = Y_hat + 1e-15  # To avoid log(0)
    loss = -np.sum(Y * np.log(Y_hat))
    return loss

def compute_cost(Y, Y_hat):
    losses = compute_crossentropy_loss(Y, Y_hat)
    cost = np.mean(losses)
    return cost

def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivatives(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def calculate_accuracy(Y, Y_hat):
    predictions = np.argmax(Y_hat, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy

def train_model(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    costs = []
    accuracies = []

    for epoch in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)

        # Compute cost
        cost = compute_cost(Y_train, A2)
        costs.append(cost)

        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, Z1, A1, Z2, A2, W2)

        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Calculate training accuracy every 10 epochs
        if epoch % 10 == 0:
            train_accuracy = calculate_accuracy(Y_train, A2)
            print(f"Epoch {epoch}/{epochs} - Cost: {cost:.4f} - Training Accuracy: {train_accuracy:.4f}")
            accuracies.append(train_accuracy)

    # Final evaluation on test set
    _, _, _, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
    test_cost = compute_cost(Y_test, A2_test)
    test_accuracy = calculate_accuracy(Y_test, A2_test)
    print(f"Final Test Cost: {test_cost:.4f} - Test Accuracy: {test_accuracy:.4f}")

    return W1, b1, W2, b2, costs, accuracies

# Load dataset
X_train, Y_train, X_test, Y_test = load_data()

# Initialize parameters
input_size = X_train.shape[0]  # 784
hidden_size = 64
output_size = 10
epochs = 100
learning_rate = 0.01

# Train the model
W1, b1, W2, b2, costs, accuracies = train_model(X_train, Y_train, X_test, Y_test,
                                                input_size, hidden_size, output_size, epochs, learning_rate)

