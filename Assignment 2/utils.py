import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist
import time

np.random.seed(0)


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function {func.__name__} took {runtime} seconds to run.")
        return result

    return wrapper


class NeuralNetwork:
    def __init__(self, model_architecture: dict[str : dict[str:int]]) -> None:
        """Initializes the neural network.

        Args:
            model_architecture (dict[str: dict[str: int]]): Neural network architecture

        Example:
        model_architecture = {
            'hidden_layer_1': {'units': 128, 'activation_function': 'relu'},
            'hidden_layer_2': {'units': 64, 'activation_function': 'relu'},
        }

        Note:
        - The last layer is the output layer with softmax activation function.
        """
        self.model_architecture = model_architecture
        self.model_state = {}
        self.number_of_images = 0
        self.train_cost = []
        self.test_cost = []
        self.train_accuracy = []
        self.test_accuracy = []

    def initialize_parameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Initializes the weights and biases of the neural network.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """

        self.number_of_images = X_train.shape[0]
        input_size = X_train.shape[1]
        n_classes = y_train.shape[1]

        # Initialize weights and biases for each hidden layer with the specified activation function.
        for layer in self.model_architecture:
            self.model_state[layer] = {}
            output_size = self.model_architecture[layer]["units"]
            self.model_state[layer]["W"] = (
                np.random.randn(output_size, input_size) * 0.01
            )
            self.model_state[layer]["b"] = np.zeros((output_size, 1))
            input_size = output_size

        # Add the output layer to the model architecture.
        self.model_architecture["output_layer"] = {
            "units": n_classes,
            "activation_function": "softmax",
        }

        # Initialize weights and biases for the output layer. Selects softmax as the activation function.
        self.model_state["output_layer"] = {}
        self.model_state["output_layer"]["W"] = (
            np.random.randn(n_classes, input_size) * 0.01
        )
        self.model_state["output_layer"]["b"] = np.zeros((n_classes, 1))

        print("Network architecture:\n")
        for layer in self.model_architecture:
            print(f"{layer}: {self.model_architecture[layer]}")
            print(f'W shape: {self.model_state[layer]["W"].shape}')
            print(f'b shape: {self.model_state[layer]["b"].shape}\n')

    def linear_forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Performs the linear part of the forward propagation.

        Args:
            X (np.ndarray): Input data
            W (np.ndarray): Weights
            b (np.ndarray): Biases

        Returns:
            Z (np.ndarray): Linear output
        """
        return np.dot(W, X) + b

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.

        Args:
            Z (np.ndarray): Linear output

        Returns:
            np.ndarray: Activation output
        """
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z: np.ndarray) -> np.ndarray:
        """ReLU activation function.

        Args:
            Z (np.ndarray): Linear output

        Returns:
            np.ndarray: Activation output
        """
        return np.maximum(0, Z)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """Softmax activation function.

        Args:
            Z (np.ndarray): Linear output

        Returns:
            np.ndarray: Activation output
        """
        exp_stable = np.exp(Z - np.max(Z))
        return exp_stable / np.sum(exp_stable, axis=0)

    def activation_forward(self, Z, activation_function: str) -> np.ndarray:
        """Performs the activation part of the forward propagation.

        Args:
            Z (np.ndarray): Linear output
            activation_function (str): Activation function

        Returns:
            A (np.ndarray): Activation output
        """
        if activation_function == "sigmoid":
            return self.sigmoid(Z)
        elif activation_function == "relu":
            return self.relu(Z)
        elif activation_function == "softmax":
            return self.softmax(Z)
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

    def model_forward(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward propagation.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Output of the last layer
        """
        feedforward_input = X.T
        for layer in self.model_state:
            # Linear forward
            self.model_state[layer]["linear_input"] = feedforward_input
            self.model_state[layer]["linear_output"] = self.linear_forward(
                feedforward_input,
                self.model_state[layer]["W"],
                self.model_state[layer]["b"],
            )

            # Activation forward
            self.model_state[layer]["activation_output"] = self.activation_forward(
                self.model_state[layer]["linear_output"],
                self.model_architecture[layer]["activation_function"],
            )

            # Update feedforward data if layer is not the output layer
            if layer != "output_layer":
                feedforward_input = self.model_state[layer]["activation_output"]
        return self.model_state["output_layer"]["activation_output"]

    def sigmoid_backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function backward pass.

        Args:
            dA (np.ndarray): Gradient of the activation output
            Z (np.ndarray): Linear output

        Returns:
            np.ndarray: Gradient of the linear output
        """
        A = self.sigmoid(Z)
        return dA * A * (1 - A)

    def relu_backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """ReLU activation function backward pass.

        Args:
            dA (np.ndarray): Gradient of the activation output
            Z (np.ndarray): Linear output

        Returns:
            np.ndarray: Gradient of the linear output
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def activation_backward(
        self, dA: np.ndarray, Z: np.ndarray, activation_function: str
    ) -> np.ndarray:
        """Performs the activation part of the backward propagation.

        Args:
            dA (np.ndarray): Gradient of the activation output
            Z (np.ndarray): Linear output
            activation_function (str): Activation function

        Returns:
            np.ndarray: Gradient of the linear output
        """
        if activation_function == "sigmoid":
            return self.sigmoid_backward(dA, Z)
        elif activation_function == "relu":
            return self.relu_backward(dA, Z)
        elif activation_function == "softmax":
            return self.sigmoid_backward(dA, Z)
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

    def linear_backward(
        self, dZ: np.ndarray, linear_input: np.ndarray, W: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs the linear part of the backward propagation.

        Args:
            dZ (np.ndarray): Gradient of the linear output
            linear_input (np.ndarray): Linear input
            W (np.ndarray): Weights

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Gradient of the input, gradient of the weights, gradient of the biases
        """
        # m = linear_input.shape[1]
        dW = np.dot(dZ, linear_input.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def model_backward(self, y: np.ndarray) -> None:
        """Performs the backward propagation.

        Args:
            y (np.ndarray): True labels
        """
        predictions = self.model_state["output_layer"]["activation_output"]
        dA = predictions - y.T

        for layer in reversed(self.model_architecture):
            dZ = self.activation_backward(
                dA,
                self.model_state[layer]["linear_output"],
                self.model_architecture[layer]["activation_function"],
            )
            dA_prev, dW, db = self.linear_backward(
                dZ,
                self.model_state[layer]["linear_input"],
                self.model_state[layer]["W"],
            )
            self.model_state[layer]["dW"] = dW
            self.model_state[layer]["db"] = db
            self.model_state[layer]["dA_prev"] = dA_prev
            dA = dA_prev

    def update_parameters(self, learning_rate: float) -> None:
        """Updates the weights and biases of the neural network.

        Args:
            learning_rate (float): Learning rate
        """
        for layer in self.model_architecture:
            self.model_state[layer]["W"] -= (
                learning_rate * self.model_state[layer]["dW"]
            )
            self.model_state[layer]["b"] -= (
                learning_rate * self.model_state[layer]["db"]
            )

    def compute_cost(self, y: np.ndarray) -> float:
        """Computes the cross-entropy cost.

        Args:
            y (np.ndarray): True labels

        Returns:
            float: Cost
        """
        predictions = self.model_state["output_layer"]["activation_output"].T
        return np.log(np.sum(np.exp(predictions), axis=1)) - np.sum(
            np.multiply(y, predictions), axis=1
        )

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predicts the class labels.

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): True labels

        Returns:
            np.ndarray: Predicted class labels
        """
        self.model_forward(X)
        predictions = np.argmax(
            self.model_state["output_layer"]["activation_output"], axis=0
        )
        classification = np.argmax(y, axis=1)
        return (predictions == classification).mean() * 100

    def random_mini_batches(
        self, X: np.ndarray, y: np.ndarray, mini_batch_size: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Creates a list of random mini-batches.

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): True labels
            mini_batch_size (int): Mini-batch size

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: List of mini-batches
        """
        mini_batches = []
        m = X.shape[0]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_y = y[permutation, :]
        n_mini_batches = m // mini_batch_size

        for i in range(n_mini_batches):
            mini_batch_X = shuffled_X[
                i * mini_batch_size : (i + 1) * mini_batch_size, :
            ]
            mini_batch_y = shuffled_y[
                i * mini_batch_size : (i + 1) * mini_batch_size, :
            ]
            mini_batches.append((mini_batch_X, mini_batch_y))

        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[n_mini_batches * mini_batch_size :, :]
            mini_batch_y = shuffled_y[n_mini_batches * mini_batch_size :, :]
            mini_batches.append((mini_batch_X, mini_batch_y))

        return mini_batches

    def plot_cost_iteration(self, iterations: int) -> None:
        """Plots the cost vs. iteration.

        Args:
            iterations (int): Number of iterations
        """
        plt.plot(range(iterations), self.train_cost, label="Train")
        plt.plot(range(iterations), self.test_cost, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost vs. Iteration")
        plt.legend()
        plt.show()

    def plot_accuracy_iteration(self, iterations: int) -> None:
        """Plots the accuracy vs. iteration.

        Args:
            iterations (int): Number of iterations
        """
        plt.plot(range(iterations), self.train_accuracy, label="Train")
        plt.plot(range(iterations), self.test_accuracy, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Iteration")
        plt.legend()
        plt.show()

    def plot_weights(self) -> None:
        """Plots the weights of the hidden_layer layer."""
        weight_matrix = self.model_state["hidden_layer_1"]["W"]
        figsize = [9.5, 5]
        nrows, ncols = 2, 5
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for counter, axi in enumerate(ax.flat):
            img = weight_matrix[counter, :].reshape((28, 28))
            axi.imshow(img)
            axi.set_title(str(counter))
        plt.show()

    @measure_runtime
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rate: float,
        mini_batch_size: int,
        iterations: int,
    ) -> None:
        """Trains the neural network.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test data
            y_test (np.ndarray): Test labels
            learning_rate (float): Learning rate
            mini_batch_size (int): Mini-batch size
            iterations (int): Number of iterations
        """
        self.initialize_parameters(X_train, y_train)
        print("Training started...")
        for i in range(iterations):
            epoch_cost = 0
            epoch_accuracy = 0
            mini_batches = self.random_mini_batches(X_train, y_train, mini_batch_size)
            for X_mini_batch, y_mini_batch in mini_batches:
                self.model_forward(X_mini_batch)
                epoch_cost += np.mean(self.compute_cost(y_mini_batch), axis=0)
                self.model_backward(y_mini_batch)
                self.update_parameters(learning_rate)
                epoch_accuracy += self.predict(X_mini_batch, y_mini_batch)
            self.train_cost.append(epoch_cost / len(mini_batches))
            self.train_accuracy.append(epoch_accuracy / len(mini_batches))
            self.test_accuracy.append(self.predict(X_test, y_test))
            self.model_forward(X_test)
            self.test_cost.append(np.mean(self.compute_cost(y_test), axis=0))

            if i % 10 == 0:
                print(f"Train Cost epoch {i}: {str(self.train_cost[i])}")
            if i % 100 == 0 and i != 0:
                print(f"Train Accuracy: {str(self.train_accuracy[i])}\n")
            elif i == (iterations - 1):
                print("Training completed!\n")
                print(f"Final Train Accuracy: {str(self.train_accuracy[i])}")
                print(f"Test Accuracy: {str(self.test_accuracy[i])}")
        self.plot_cost_iteration(iterations)
        self.plot_accuracy_iteration(iterations)


def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    model_architecture = {
        "hidden_layer_1": {"units": 128, "activation_function": "relu"},
        "hidden_layer_2": {"units": 64, "activation_function": "relu"},
    }
    neural_network = NeuralNetwork(model_architecture)
    neural_network.initialize_parameters(X_train, y_train)
    # neural_network.train_model(X_train, y_train, X_test, y_test, 0.01, 64, 1000)


if __name__ == "__main__":
    main()
