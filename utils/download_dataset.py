from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def download_dataset(root, isDownload, batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        # 1. Helps the model to converge faster
        # 2. Helps to make the numerical computations stable
    ])

    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=isDownload,
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=isDownload,
        transform=transform
    )

    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    
    print(f"DataLoader: {train_loader, test_loader}")
    print(f"Length of train_loader: {len(train_loader)} batches of {batch_size}...")
    print(f"Length of test_loader: {len(test_loader)} batches of {batch_size}...")


    return train_loader, test_loader