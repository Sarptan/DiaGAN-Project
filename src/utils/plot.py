import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import torch


def plot_batch_images(images, labels):
    batch_size = images.shape[0]
    nrow = int(np.sqrt(batch_size))
    grid_img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)  # Arrange images in grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.axis("off")
    plt.title("MNIST Batch Samples")
    plt.show()


import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch

def visualize_colored_mnist(dataloader, title="Colored MNIST Batch Samples"):
    images, labels = next(iter(dataloader))

    # Convert from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    batch_size = images.shape[0]
    nrow = int(np.sqrt(batch_size))

    # Make grid without normalization (we already did it)
    grid_img = torchvision.utils.make_grid(images, nrow=nrow, padding=2)

    # Convert to numpy and permute to (H, W, C)
    np_grid = grid_img.permute(1, 2, 0).cpu().numpy()

    # Plot with no interpolation and larger figsize
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid, interpolation='none')  # <- disables blending
    plt.axis("off")
    plt.title(title)
    plt.show()

