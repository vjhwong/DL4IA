import torch
import imageio
import glob
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def load_warwick(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the WARWICK dataset.

    Args:
        device (torch.device): Device to load the dataset on

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Training and test images and labels
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Load images and labels
    for image_path in sorted(
        glob.glob("../data/WARWICK/WARWICK" + "/Train/image_*.png")
    ):
        image = imageio.imread(image_path)
        train_images.append(image)
    for label_path in sorted(
        glob.glob("../data/WARWICK/WARWICK" + "/Train/label_*.png")
    ):
        label = imageio.imread(label_path)
        train_labels.append(label)

    for image_path in sorted(
        glob.glob("../data/WARWICK/WARWICK" + "/Test/image_*.png")
    ):
        image = imageio.imread(image_path)
        test_images.append(image)
    for label_path in sorted(
        glob.glob("../data/WARWICK/WARWICK" + "/Test/label_*.png")
    ):
        label = imageio.imread(label_path)
        test_labels.append(label)

    # Normalize images and labels
    X_train = torch.tensor(
        np.array(train_images), dtype=torch.float, requires_grad=True
    ).permute(0, 3, 1, 2)
    Y_train = torch.tensor(np.array(train_labels), dtype=torch.long) / 255
    X_test = torch.tensor(
        np.array(test_images), dtype=torch.float, requires_grad=True
    ).permute(0, 3, 1, 2)
    Y_test = torch.tensor(np.array(test_labels), dtype=torch.long) / 255

    return X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)


def dice_coefficient(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the Dice coefficient.

    Args:
        outputs (torch.Tensor): Output of the model
        targets (torch.Tensor): Target labels

    Returns:
        torch.Tensor: Dice coefficient
    """
    # Flatten outputs and targets
    batch_size = outputs.size(0)
    outputs_flat = outputs.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)

    # Calculate intersection and union
    intersection = (outputs_flat * targets_flat).sum(1)
    unionset = outputs_flat.sum(1) + targets_flat.sum(1)

    # Calculate Dice coefficient
    dice = 2 * (intersection) / (unionset + 1e-8)
    return dice.sum() / batch_size


def training_curve_plot(
    title: str,
    train_losses: list[float],
    test_losses: list[float],
    train_accuracy: list[float],
    test_accuracy: list[float],
) -> None:
    """Plot the training curve.

    Args:
        title (str): Title of the plot
        train_losses (list[float]): Training losses
        test_losses (list[float]): Test losses
        train_accuracy (list[float]): Training accuracy
        test_accuracy (list[float]): Test accuracy
    """
    md = 10
    sm = 9
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=md)

    # Plot the training and test losses
    x = range(1, len(train_losses) + 1)
    axs[0].plot(x, train_losses, label=f"Final train cost: {train_losses[-1]:.4f}")
    axs[0].plot(x, test_losses, label=f"Final test cost: {test_losses[-1]:.4f}")
    axs[0].set_title("Cost", fontsize=md)
    axs[0].set_xlabel("Epoch", fontsize=md)
    axs[0].set_ylabel("Cost", fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis="both", labelsize=sm)
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Plot the training and test accuracy
    axs[1].plot(
        x, train_accuracy, label=f"Final train Dice score: {train_accuracy[-1]:.4f}"
    )
    axs[1].plot(
        x, test_accuracy, label=f"Final test Dice score: {test_accuracy[-1]:.4f}"
    )
    axs[1].set_title("Dice score", fontsize=md)
    axs[1].set_xlabel("Epoch", fontsize=md)
    axs[1].set_ylabel("Dice", fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis="both", labelsize=sm)
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def train_segmentation_network(
    model: nn.Module,
    criteria: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> None:
    """
    Train a neural network model for segmentation

    Args:
        model: The neural network model to be trained
        criteria: The loss function
        optimizer: The optimizer
        num_epochs: The number of epochs to train the model
        train_loader: The training data loader
        test_loader: The test data loader
        device: The device to run the model on
    """
    train_losses = []
    test_losses = []
    train_dice_scores = []
    test_dice_scores = []
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0

        # Train the model
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criteria(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Calculate Dice score
            outputs = (outputs > 0.5).float()
            dice = dice_coefficient(outputs, labels)
            train_dice += dice.item()

        train_losses.append(train_loss / len(train_loader))
        train_dice_scores.append(train_dice / len(train_loader))

        model.eval()
        test_loss = 0
        test_dice = 0

        # Test the model
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criteria(outputs, labels)
                test_loss += loss.item()

                # Calculate Dice score
                outputs = (outputs > 0.5).float()
                dice = dice_coefficient(outputs, labels)
                test_dice += dice.item()

            test_losses.append(test_loss / len(test_loader))
            test_dice_scores.append(test_dice / len(test_loader))

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Dice Score: {train_dice_scores[-1]:.4f}, Test Dice Score: {test_dice_scores[-1]:.4f}"
        )
    runtime = time.time() - start_time
    plot_title = f"{model_name}\nLearning rate: {optimizer.param_groups[0]['lr']} - Batch size: {train_loader.batch_size} - Optimizer: {type(optimizer).__name__}- Total Time: {runtime//60:.0f} min {runtime%60:.2f} s"
    training_curve_plot(
        plot_title, train_losses, test_losses, train_dice_scores, test_dice_scores
    )


def find_best_and_worst_case(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor | float | int], dict[str, torch.Tensor | float | int]
]:
    """Find the best and worst case in the test set.

    Args:
        model (torch.nn.Module): Model to evaluate
        X_test (torch.Tensor): Test images
        Y_test (torch.Tensor): Test labels
        device (torch.device): Device to run the model on

    Returns:
        tuple[ dict[str, torch.Tensor | float | int], dict[str, torch.Tensor | float | int] ]: Best and worst case
    """
    best_case = {
        "input": None,
        "label": None,
        "output": None,
        "performance": float("-inf"),
        "index": None,
    }
    worst_case = {
        "input": None,
        "label": None,
        "output": None,
        "performance": float("inf"),
        "index": None,
    }
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(zip(X_test, Y_test)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = (outputs > 0.5).float()
            performance = dice_coefficient(outputs, labels)

            # Update best or worst case
            if performance > best_case["performance"]:
                best_case["input"] = inputs
                best_case["label"] = labels
                best_case["output"] = outputs
                best_case["performance"] = performance
                best_case["index"] = index

            elif performance < worst_case["performance"]:
                worst_case["input"] = inputs
                worst_case["label"] = labels
                worst_case["output"] = outputs
                worst_case["performance"] = performance
                worst_case["index"] = index
    return best_case, worst_case


def visualize_case(
    case: dict[str : torch.Tensor | float | int], device: torch.device, name: str
) -> None:
    """Visualize a case.

    Args:
        case (dict[str : torch.Tensor | float | int]): Case to visualize
        device (torch.device): Device to run the model on
        name (str): Name of the case
    """
    fontsize = 16

    # Convert tensors to numpy arrays
    image = case["input"].detach().cpu().permute(1, 2, 0).numpy().astype("uint8")
    mask = case["label"].detach().cpu()
    prediction = (
        case["output"].unsqueeze(0).to(device).detach().cpu().numpy().astype("uint8")
    )

    # Plot the image, mask, and prediction
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{name} - Dice score: {case['performance']:.4f}\n", fontsize=fontsize)

    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=fontsize)
    axes[0].axis("off")

    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth Mask", fontsize=fontsize)
    axes[1].axis("off")

    axes[2].imshow(prediction.squeeze(), cmap="gray")
    axes[2].set_title("Predicted Mask", fontsize=fontsize)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
