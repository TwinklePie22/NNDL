import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for xi, yi in zip(x, y):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                error = yi - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict(self, x):
        results = []
        for xi in x:
            z = np.dot(xi, self.weights) + self.bias
            results.append(1 if self.sigmoid(z) >= 0.5 else 0)
        return results

# Input data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND gate outputs
y_or = np.array([0, 1, 1, 1])   # OR gate outputs

# Train and test for AND gate
perceptron_and = Perceptron()
perceptron_and.fit(x, y_and)
print("AND gate predictions:")
for xi, pred in zip(x, perceptron_and.predict(x)):
    print(f"Input: {xi} - Prediction: {pred}")

# Train and test for OR gate
perceptron_or = Perceptron()
perceptron_or.fit(x, y_or)
print("\nOR gate predictions:")
for xi, pred in zip(x, perceptron_or.predict(x)):
    print(f"Input: {xi} - Prediction: {pred}")
