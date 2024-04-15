import numpy as np
from load_mnist import load_mnist


class NeuralNetwork:
    def __init__(self, model):
        self.model = model
        self.parameters = {}
        self.model_state = {}

    def initialize_parameters(self):
        np.random.seed(0)
        for i in range(1, len(self.model)):
            self.parameters["W" + str(i)] = (
                np.random.randn(self.model[i], self.model[i - 1]) * 0.01
            )
            self.parameters["b" + str(i)] = np.zeros((self.model[i], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
        activation_cache = Z
        cache = (linear_cache, activation_cache)
        return A, cache

    def model_forward(self, X):
        A = X
        self.model_state["A0"] = A
        for i in range(1, len(self.model)):
            A_prev = A
            A, cache = self.activation_forward(
                A_prev,
                self.parameters["W" + str(i)],
                self.parameters["b" + str(i)],
                "relu",
            )
            self.model_state["A" + str(i)] = A
            self.model_state["cache" + str(i)] = cache
        return A

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z))
        return exp_Z / np.sum(exp_Z, axis=0)

    def compute_loss(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL)) / m
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def sigmoid_backward(self, dA, Z):
        A = self.sigmoid(Z)
        dZ = dA * A * (1 - A)
        return dZ

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        elif activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def model_backward(self, AL, Y):
        grads = {}
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        m = Y.shape[1]
        (
            grads["dA" + str(len(self.model) - 1)],
            grads["dW" + str(len(self.model) - 1)],
            grads["db" + str(len(self.model) - 1)],
        ) = self.activation_backward(
            dAL, self.model_state["cache" + str(len(self.model) - 1)], "sigmoid"
        )
        for i in reversed(range(len(self.model) - 1)):
            current_cache = self.model_state["cache" + str(i)]
            grads["dA" + str(i)], grads["dW" + str(i)], grads["db" + str(i)] = (
                self.activation_backward(
                    grads["dA" + str(i + 1)], current_cache, "relu"
                )
            )
        return grads

    def update_parameters(self, grads, learning_rate):
        for i in range(1, len(self.model)):
            self.parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
            self.parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]

    def predict(self, X):
        AL = self.model_forward(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

    def random_mini_batches(self, X, Y, batch_size):
        m = X.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        num_complete_minibatches = m // batch_size
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * batch_size : (k + 1) * batch_size]
            mini_batch_Y = shuffled_Y[:, k * batch_size : (k + 1) * batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size :]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * batch_size :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def train_model(
        self,
        x_train,
        y_train,
        iterations,
        learning_rate,
        batch_size,
        x_test=None,
        y_test=None,
        k=100,
    ):
        train_costs = []
        train_accuracies = []
        test_costs = []
        test_accuracies = []
        for i in range(iterations):
            mini_batches = self.random_mini_batches(x_train, y_train, batch_size)
            for mini_batch in mini_batches:
                mini_batch_X, mini_batch_Y = mini_batch
                AL = self.model_forward(mini_batch_X)
                cost = self.compute_loss(AL, mini_batch_Y)
                grads = self.model_backward(AL, mini_batch_Y)
                self.update_parameters(grads, learning_rate)
            if i % k == 0:
                train_AL = self.model_forward(x_train)
                train_cost = self.compute_loss(train_AL, y_train)
                train_accuracy = np.mean(
                    np.argmax(train_AL, axis=0) == np.argmax(y_train, axis=0)
                )
                train_costs.append(train_cost)
                train_accuracies.append(train_accuracy)
                if x_test is not None and y_test is not None:
                    test_AL = self.model_forward(x_test)
                    test_cost = self.compute_loss(test_AL, y_test)
                    test_accuracy = np.mean(
                        np.argmax(test_AL, axis=0) == np.argmax(y_test, axis=0)
                    )
                    test_costs.append(test_cost)
                    test_accuracies.append(test_accuracy)
        return train_costs, train_accuracies, test_costs, test_accuracies


X_train, y_train, X_test, y_test = load_mnist()
model = NeuralNetwork([784, 128, 10])
model.initialize_parameters()
train_costs, train_accuracies, test_costs, test_accuracies = model.train_model(
    X_train,
    y_train,
    iterations=1000,
    learning_rate=0.01,
    batch_size=64,
    x_test=X_test,
    y_test=y_test,
    k=100,
)
