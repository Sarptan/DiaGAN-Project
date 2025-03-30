import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import torch.nn as nn 

def load_MNISTdata():
    data_dir = os.path.join(os.getcwd(), "..", "data", "raw")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

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
        self.images, self.labels = mnist_data.data, mnist_data.targets  # Extract images n labels
        self.images = self.images.unsqueeze(1).float() / 255.0

        total_samples = len(self.images)
        num_minority = int(total_samples * minority_ratio)  # Define the number of green samples

        indices = torch.randperm(total_samples)  

        # Create empty color channels (N, 3, 28, 28)
        self.colors = torch.zeros((total_samples, 3, 28, 28))  

        # Majority samples: Red (R,0,0)
        self.colors[indices[num_minority:], 0] = self.images[indices[num_minority:]].squeeze(1)  # Red channel

        # Minority samples: Green (0,G,0)
        self.colors[indices[:num_minority], 1] = self.images[indices[:num_minority]].squeeze(1)  # Green channel

        self.colors = (self.colors - 0.5) * 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.colors[idx], self.labels[idx]
        return img, label
    

def get_colored_mnist_dataloader(mnist_dataset, batch_size=128, minority_ratio=0.1):
    dataset = ColoredMNIST(mnist_dataset, minority_ratio=minority_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def generate_gaussian_modes(n_samples_per_mode=500, seed=42):
    np.random.seed(seed)

    # 5x5 grid of Gaussian centers
    centers = [(x, y) for x in range(5) for y in range(5)]  # (0,0) to (4,4)
    data = []
    labels = []

    for i, (cx, cy) in enumerate(centers):
        # Assign red for majority, green for minority

        points = np.random.randn(n_samples_per_mode, 2) * 0.05 + np.array([cx, cy])
        data.append(points)
        labels += [i] * n_samples_per_mode

    data = np.vstack(data).astype(np.float32)
    labels = np.array(labels)
    return data, labels


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 → 14x14

            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits, feats.view(x.size(0), -1)  # logits, feature_vector


def load_mnist_classifier(path="mnist_classifier.pth"):
    model = MNISTClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


def loader_to_data(train_loader):
    all_imgs = []
    for imgs, _ in train_loader:
        all_imgs.append(imgs)
    data = torch.cat(all_imgs, dim=0).numpy()  
    print(f"Loaded: {data.shape}")
    return data


import torch.nn as nn

class ColoredMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 → 14x14

            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits, feats.view(x.size(0), -1)  # logits, feature_vector


def load_colored_mnist_classifier(path="colored_mnist_classifier.pth"):
    model = ColoredMNISTClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
