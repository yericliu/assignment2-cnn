from torchvision import transforms


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Use the same preprocessing during training and API prediction.
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_data_loader(data_path, batch_size=64, train=True):
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=data_path, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
