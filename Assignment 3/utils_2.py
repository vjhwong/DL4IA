import torch
import torch.nn as nn
import numpy as np
import imageio
import glob


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

    X_train = torch.tensor(
        np.array(train_images), dtype=torch.float, requires_grad=True
    ).permute(0, 3, 1, 2)
    Y_train = torch.tensor(np.array(train_labels), dtype=torch.long) / 255
    X_test = torch.tensor(
        np.array(test_images), dtype=torch.float, requires_grad=True
    ).permute(0, 3, 1, 2)
    Y_test = torch.tensor(np.array(test_labels), dtype=torch.long) / 255

    return X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
