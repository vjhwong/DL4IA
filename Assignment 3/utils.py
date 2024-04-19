import matplotlib.pyplot as plt


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
