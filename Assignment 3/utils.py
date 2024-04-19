import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_learning_curves(
    num_epochs: int,
    train_losses: list[int],
    test_losses: list[int],
    train_accuracies: list[int],
    test_accuracies: list[int],
) -> None:
    """Plot the learning curves for the training and test losses and accuracies.

    Args:
        num_epochs (int): Number of epochs
        train_losses (list[int]): List of training losses
        test_losses (list[int]): List of test losses
        train_accuracies (list[int]): List of training accuracies
        test_accuracies (list[int]): List of test accuracies
    """
    plt.subplot(2, 1, 1)
    plt.plot(
        range(1, num_epochs + 1),
        train_losses,
        label=f"Final train loss: {train_losses[-1]}",
    )
    plt.plot(
        range(1, num_epochs + 1),
        test_losses,
        label=f"Final test loss: {test_losses[-1]}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        range(1, num_epochs + 1),
        train_accuracies,
        label=f"Final train accuracy: {train_accuracies[-1]}",
    )
    plt.plot(
        range(1, num_epochs + 1),
        test_accuracies,
        label=f"Final test accuracy: {test_accuracies[-1]}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_neural_network(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
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
        for i, (images, labels) in enumerate(train_loader):
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

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

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
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_costs[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Test Loss: {test_costs[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%"
        )

    plot_learning_curves(
        num_epochs, train_costs, test_costs, train_accuracies, test_accuracies
    )
