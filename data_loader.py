import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from pathlib import Path

# 这是一个辅助类，专门用于对数据集子集（Subset）应用变换
class ApplyTransform(Dataset):
    """
    为一个数据集（特别是 Subset）应用指定的 transform。
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        transformed_sample = self.transform(sample)
        return transformed_sample, label

def get_dataloaders(dataset_root_path, batch_size=32):
    """
    创建并返回训练和测试的 DataLoader。

    参数:
        dataset_root_path (str or Path): 数据集根目录的路径。
        batch_size (int): 每个批次的大小。

    返回:
        tuple: (train_loader, test_loader)
    """
    HEIGHT = 224
    WIDTH = 224

    mean = [0.4883356988430023, 0.45529747009277344, 0.4170495867729187]
    std = [0.2598116099834442, 0.25313833355903625, 0.255852073431015]

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((HEIGHT, WIDTH)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((HEIGHT, WIDTH)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    dataset_root = Path(dataset_root_path)
    full_dataset = datasets.ImageFolder(root=dataset_root)

    # 分割数据集
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # 为分割后的子集应用不同的变换
    train_dataset = ApplyTransform(train_subset, transform=transform_train)
    test_dataset = ApplyTransform(test_subset, transform=transform_test)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader