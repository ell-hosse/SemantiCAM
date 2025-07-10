import torch
from torchvision import transforms
from medmnist import PathMNIST
from config import IMAGE_SIZE, BATCH_SIZE

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    train_ds = PathMNIST(split='train', transform=transform, download=True)
    test_ds = PathMNIST(split='test', transform=transform, download=True)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )
