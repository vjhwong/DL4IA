import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import imageio.v2 as imageio
import glob


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime of {func.__name__}: {runtime:.4f} seconds")
        return result

    return wrapper


def plot_learning_curves(
    num_epochs: int,
    train_losses: list[int],
    test_losses: list[int],
    train_accuracies: list[int],
    test_accuracies: list[int],
    title: str,
) -> None:
    """Plot the learning curves for the training and test losses and accuracies.

    Args:
        num_epochs (int): Number of epochs
        train_losses (list[int]): List of training losses
        test_losses (list[int]): List of test losses
        train_accuracies (list[int]): List of training accuracies
        test_accuracies (list[int]): List of test accuracies
    """
    lg = 13
    md = 10
    sm = 9
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=lg)

    axs[0].plot(
        range(1, num_epochs + 1),
        train_losses,
        label=f"Final train cost: {train_losses[-1]:.4f}",
    )
    axs[0].plot(
        range(1, num_epochs + 1),
        test_losses,
        label=f"Final test cost: {test_losses[-1]:.4f}",
    )
    axs[0].set_xlabel("Epoch", fontsize=md)
    axs[0].set_ylabel("Cost", fontsize=md)
    axs[0].set_title("Cost", fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(labelsize=sm)
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    axs[1].plot(
        range(1, num_epochs + 1),
        train_accuracies,
        label=f"Final train accuracy: {train_accuracies[-1]:.4f}%",
    )
    axs[1].plot(
        range(1, num_epochs + 1),
        test_accuracies,
        label=f"Final test accuracy: {test_accuracies[-1]:.4f}%",
    )
    axs[1].set_xlabel("Epoch", fontsize=md)
    axs[1].set_ylabel("Accuracy (%)", fontsize=md)
    axs[1].set_title("Accuracy", fontsize=md)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(labelsize=sm)
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()


@measure_runtime
def train_neural_network(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    model_name: str,
) -> None:
    """Train a neural network model and evaluate it on a test set.

    Args:
        model (nn.Module): Model to train
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of epochs
        device (torch.device): Device to run the model on
    """
    start = time.time()
    train_costs = []
    train_accuracies = []
    test_costs = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for images, labels in train_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            train_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()

        train_costs.append(train_loss / len(train_loader))
        train_accuracies.append(100 * (correct_train / total_train))

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        # Validation loop
        with torch.no_grad():
            for images, labels in test_loader:
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == torch.argmax(labels, dim=1)).sum().item()

            test_costs.append(test_loss / len(test_loader))
            test_accuracies.append(100 * (correct_test / total_test))

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_costs[-1]:.4f}, Test Loss: {test_costs[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Test Accuracy: {test_accuracies[-1]:.2f}%"
        )
    end_time = time.time()
    title = f"{model_name}\nLearning rate: {optimizer.param_groups[0]['lr']}, Batch size: {train_loader.batch_size}, Optimizer: {type(optimizer).__name__}, Time: {end_time - start:.2f} seconds"
    plot_learning_curves(
        num_epochs, train_costs, test_costs, train_accuracies, test_accuracies, title
    )


def compute_confusion_matrix(predictions: torch.Tensor, true_labels: torch.Tensor):
    """Compute the confusion matrix for a set of predictions and true labels.

    Args:
        predictions (torch.Tensor): Predictions
        true_labels (torch.Tensor): True labels

    Returns:
        torch.Tensor: Confusion matrix
    """
    predictions = predictions.argmax(dim=1)
    true_labels = true_labels.argmax(dim=1)
    matrix = torch.zeros(10, 10, dtype=torch.int32).to(predictions.device)
    for datapoint in range(len(true_labels)):
        prediction = predictions[datapoint]
        true_class = true_labels[datapoint]
        matrix[prediction, true_class] += 1
    return matrix


def find_misclassified_indices(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """Find the indices of misclassified images in a test set.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run the model on

    Returns:
        torch.Tensor: Indices of misclassified images
    """
    misclassified_indices = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            misclassified_indices.extend(
                (predicted != torch.argmax(labels, dim=1)).nonzero().squeeze().tolist()
            )
    return torch.tensor(misclassified_indices)


def load_warwick():

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for image_path in sorted(glob.glob("WARWICK/WARWICK" + "/Train/image_*.png")):
        image = imageio.imread(image_path)
        train_images.append(image)
    for label_path in sorted(glob.glob("WARWICK/WARWICK" + "/Train/label_*.png")):
        label = imageio.imread(label_path)
        train_labels.append(label)

    for image_path in sorted(glob.glob("WARWICK/WARWICK" + "/Test/image_*.png")):
        image = imageio.imread(image_path)
        test_images.append(image)
    for label_path in sorted(glob.glob("WARWICK/WARWICK" + "/Test/label_*.png")):
        label = imageio.imread(label_path)
        test_labels.append(label)

    X_train = torch.tensor(
        np.array(train_images), dtype=torch.float, requires_grad=True
    ).permute(0, 3, 1, 2)
    Y_train = torch.tensor(np.array(train_labels), dtype=torch.long) / 255
    X_test = torch.tensor(
        np.array(test_images), dtype=torch.float, requires_grad=True
    ).permute(0, 3, 1, 2)
    Y_test = torch.tensor(np.array(test_labels), dtype=torch.long) / 255

    return X_train, Y_train, X_test, Y_test
