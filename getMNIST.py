import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

# Define paths
data_dir = os.path.join(os.getcwd(), "data", "raw")  # Path to store raw MNIST data
os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists

# Download MNIST dataset
mnist_train = datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)

print(f"MNIST dataset downloaded successfully to: {data_dir}")