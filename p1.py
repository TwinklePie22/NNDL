import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def train_perceptron(x, y, w1, w2, bias, learning_rate, epochs=500000):
    for epoch in range(epochs):
        for i in range(4):
            z = x[i][0] * w1 + x[i][1] * w2 + bias
            result = 1 / (1 + np.exp(-z))
            error = y[i] - result
            w1 += learning_rate * error * x[i][0]
            w2 += learning_rate * error * x[i][1]
            bias += learning_rate * error
    return w1, w2, bias

def test_perceptron(x, w1, w2, bias):
    for i in range(4):
        z = x[i][0] * w1 + x[i][1] * w2 + bias
        result = 1 / (1 + np.exp(-z))
        print(f"Input: {x[i]}, Output: {result:.4f}, Predicted: {1 if result >= 0.5 else 0}")

# AND gate
y_and = np.array([0, 0, 0, 1])
w1, w2, bias = train_perceptron(x, y_and, 0.8, 0.9, 0.25, 0.1)
print("AND gate results:")
test_perceptron(x, w1, w2, bias)

# OR gate
y_or = np.array([0, 1, 1, 1])
w1, w2, bias = train_perceptron(x, y_or, 0.8, 0.9, 0.25, 0.1)
print("\nOR gate results:")
test_perceptron(x, w1, w2, bias)
