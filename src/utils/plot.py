import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np


def plot_batch_images(images, labels):
    batch_size = images.shape[0]
    nrow = int(np.sqrt(batch_size))
    grid_img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)  # Arrange images in grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.axis("off")
    plt.title("MNIST Batch Samples")
    plt.show()


def visualize_colored_mnist(dataloader, title="Colored MNIST Batch Samples"):
    """
    Visualizes a batch of Colored MNIST images using a grid layout.

    Args:
        dataloader (DataLoader): DataLoader containing Colored MNIST samples.
        title (str): Title of the plot.
    """
    images, labels = next(iter(dataloader))  # Get a batch
    batch_size = images.shape[0]
    nrow = int(np.sqrt(batch_size))  # Arrange images in a square grid

    # Create an image grid
    grid_img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)

    # Plot the image batch
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.axis("off")
    plt.title(title)
    plt.show()