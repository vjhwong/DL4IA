import numpy as np
from matplotlib import axes


def plot_cost_history(learning_rate: float, cost_history: np.ndarray, ax: axes) -> None:
    """Plots the cost history on the given axes.

    Args:
        cost_history (np.ndarray): Array of cost values.
        ax (axes): Axes object to plot on.
    """
    (line,) = ax.plot(cost_history)
    line.set_label(f"$\\alpha$ = 1e{learning_rate}")


def initialize_parameters(n_x: int, n_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Initializes the weights and biases of the neural network.

    Args:
        n_x (int): Number of input features.
        n_y (int): Number of output classes.

    Returns:
        tuple[np.ndarray, np.ndarray]: The weights and biases of the neural network.
    """
    w = np.zeros((n_x, n_y))
    b = np.random.randn(n_y)
    return w, b


def model_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward pass of the neural network.

    Args:
        x (np.ndarray): Features of the dataset.
        w (np.ndarray): Weights of the neural network.
        b (np.ndarray): Biases of the neural network.

    Returns:
        np.ndarray: The output of the neural network.
    """
    return np.dot(x, w) + b


def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Computes the cost of the neural network.

    Args:
        y (np.ndarray): True labels of the dataset.
        y_hat (np.ndarray): Predicted labels of the dataset.

    Returns:
        float: The cost of the neural network.
    """
    return np.mean((y - y_hat) ** 2)


def model_backward(
    x: np.ndarray, y: np.ndarray, y_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backward pass of the neural network.

    Args:
        x (np.ndarray): Features of the dataset.
        y (np.ndarray): True labels of the dataset.
        y_hat (np.ndarray): Predicted labels of the dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: The gradients of the weights and biases.
    """
    m = x.shape[0]
    dw = -2 / m * np.dot(x.T, (y - y_hat))
    db = -2 / m * np.sum(y - y_hat)
    return dw, db


def update_parameters(
    w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, learning_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Updates the weights and biases of the neural network.

    Args:
        w (np.ndarray): Weights of the neural network.
        b (np.ndarray): Biases of the neural network.
        dw (np.ndarray): Gradients of the weights.
        db (np.ndarray): Gradients of the biases.
        learning_rate (float): Learning rate of the neural network.

    Returns:
        tuple[np.ndarray, np.ndarray]: The updated weights and biases.
    """
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b


def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Predicts zi based on the corresponding input xi.

    Args:
        x (np.ndarray): Input features.
        w (np.ndarray): Weights of the neural network.
        b (np.ndarray): Biases of the neural network.

    Returns:
        np.ndarray: Predicted values.
    """
    return model_forward(x, w, b)


def train_linear_model(
    x: np.ndarray, y: np.ndarray, num_iterations: int, learning_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trains a linear model using gradient descent.

    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): True labels.
        num_iterations (int): Number of iterations for training.
        learning_rate (float): Learning rate for gradient descent.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized parameters of the model and cost history.
    """
    # Initialize parameters
    w, b = initialize_parameters(x.shape[1], 1)

    # Initialize cost vector
    cost_history = np.empty(num_iterations)

    # Perform gradient descent
    for i in range(num_iterations):
        # Forward pass
        y_hat = model_forward(x, w, b)

        # Compute cost
        cost_history[i] = compute_cost(y, y_hat)

        # Backward pass
        dw, db = model_backward(x, y, y_hat)

        # Update parameters
        w, b = update_parameters(w, b, dw, db, learning_rate)

    return w, b, cost_history
