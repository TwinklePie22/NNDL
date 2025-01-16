import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_perceptron(x, y, w1, w2, bias, learning_rate, epochs=500000):
    for epoch in range(epochs):
        for i in range(4):
            z = x[i][0] * w1 + x[i][1] * w2 + bias
            result = sigmoid(z)
            error = y[i] - result
            w1 += learning_rate * error * x[i][0]
            w2 += learning_rate * error * x[i][1]
            bias += learning_rate * error
    return w1, w2, bias

def test_perceptron(x, w1, w2, bias):
    for i in range(4):
        z = x[i][0] * w1 + x[i][1] * w2 + bias
        result = sigmoid(z)
        print(f"Input: {x[i]}, Output: {result:.4f}, Predicted: {1 if result >= 0.5 else 0}")

# Initial weights and bias
w1, w2, bias = 0.8, 0.9, 0.25
learning_rate = 0.1

# Train and test for AND gate
y_and = np.array([0, 0, 0, 1])
w1, w2, bias = train_perceptron(x, y_and, w1, w2, bias, learning_rate)
print("AND gate results:")
test_perceptron(x, w1, w2, bias)

# Train and test for OR gate
y_or = np.array([0, 1, 1, 1])
w1, w2, bias = train_perceptron(x, y_or, w1, w2, bias, learning_rate)
print("\nOR gate results:")
test_perceptron(x, w1, w2, bias)
