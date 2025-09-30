import torch
import torchvision.transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

folder = './CatADog_5340'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()
])

dataset_1 = ImageFolder(folder, transform=transform)

dataLoader = DataLoader(
    dataset_1,
    batch_size=32,
    num_workers=0,
    shuffle=False
)

mean = torch.zeros(3)
std = torch.zeros(3)
total_pixels = 0

for batch_idx, (images, _) in enumerate(dataLoader):
    print(f"处理批次 {batch_idx + 1}/{len(dataLoader)}")
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    total_pixels += images.size(2) * batch_samples
    
 # 计算整体均值
mean /= len(dataLoader.dataset)
    
# 计算标准差
for batch_idx, (images, _) in enumerate(dataLoader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    
    # 累加方差
    std += ((images - mean.view(1, 3, 1)) ** 2).mean(2).sum(0)

# 计算整体标准差
std = torch.sqrt(std / len(dataLoader.dataset))
mean,std = mean.tolist(), std.tolist()

print(f"均值 (RGB): {mean}")
print(f"标准差 (RGB): {std}")

# 均值 (RGB): [0.4883356988430023, 0.45529747009277344, 0.4170495867729187]
# 标准差 (RGB): [0.2598116099834442, 0.25313833355903625, 0.255852073431015]