import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def load_MNISTdata():
    data_dir = os.path.join(os.getcwd(), "..", "data", "raw")  # Move up from notebooks/

    # Define transformation (convert images to tensors)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=False)

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


class ColoredMNIST(Dataset):
    def __init__(self, mnist_data, minority_ratio=0.1):
        """
        Custom Colored MNIST dataset that assigns red to majority and green to a specified minority group.

        Args:
            mnist_data (torchvision.datasets.MNIST): Preloaded MNIST dataset.
            minority_ratio (float): Fraction of samples to be colored green (0 to 1).
        """
        super().__init__()
        self.images, self.labels = mnist_data.data, mnist_data.targets  # Extract images & labels
        self.images = self.images.unsqueeze(1).float() / 255.0  # Normalize to [0,1]

        total_samples = len(self.images)
        num_minority = int(total_samples * minority_ratio)  # Define the number of green samples

        indices = torch.randperm(total_samples)  # Shuffle indices

        # Create empty color channels (N, 3, 28, 28)
        self.colors = torch.zeros((total_samples, 3, 28, 28))  

        # Majority samples: Red (R,0,0)
        self.colors[indices[num_minority:], 0] = self.images[indices[num_minority:]].squeeze(1)  # Red channel

        # Minority samples: Green (0,G,0)
        self.colors[indices[:num_minority], 1] = self.images[indices[:num_minority]].squeeze(1)  # Green channel

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.colors[idx], self.labels[idx]
        return img, label
    

def get_colored_mnist_dataloader(mnist_dataset, batch_size=128, minority_ratio=0.1):
    dataset = ColoredMNIST(mnist_dataset, minority_ratio=minority_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader