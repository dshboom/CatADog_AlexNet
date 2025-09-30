import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from PIL import Image
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

HEIGHT = 224
WIDTH = 224

mean = [0.4883356988430023, 0.45529747009277344, 0.4170495867729187]
std = [0.2598116099834442, 0.25313833355903625, 0.255852073431015]
# 你可以把这个类加到你的脚本顶部
class ApplyTransform(torch.utils.data.Dataset):
    """
    为一个数据集（特别是 Subset）应用指定的 transform。
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从原始数据集中获取样本
        sample, label = self.dataset[idx]
        # 应用指定的 transform
        transformed_sample = self.transform(sample)
        return transformed_sample, label
    

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((HEIGHT, WIDTH)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),     
    torchvision.transforms.Pad(padding=0),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean,std=std),
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((HEIGHT, WIDTH)),
    torchvision.transforms.Pad(padding=0),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean,std=std)
])

# 2. 使用 pathlib 来创建健壮的路径对象
dataset_root = Path('./CatADog_5340') 

# 3. 将 Path 对象传递给 ImageFolder
dataset = datasets.ImageFolder(root=dataset_root)

train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size
train_subset, test_subset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
train_dataset = ApplyTransform(train_subset, transform=transform_train)
test_dataset = ApplyTransform(test_subset, transform=transform_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True    
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# print("类别名称:", dataset.classes) 类别名称: ['Cat', 'Dog']

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=2
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),            
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = AlexNet(num_classes=2)

# dummy_input = torch.randn(1, 3, 224, 224)
# output = model(dummy_input)
# print(f"带有BN的AlexNet输出尺寸: {output.shape}") # 应该输出 torch.Size([1, 2])

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
criterion = nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=3
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

num_epoch = 50
patience = 5
epoch_no_improve = 0
best_val_acc = 0.0

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss:.4f}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epoch}] -> Train Loss: {epoch_loss:.4f} | Val Acc: {accuracy:.2f}% | LR: {current_lr:.6f}")
        scheduler.step(accuracy)
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            epoch_no_improve = 0
            torch.save(
                model.state_dict(),
                'best_alexnet_model_CatADog.pth'
            )
            print(f"  -> Val Acc Improved to {best_val_acc:.2f}%. Saving model...")
        else:
            epoch_no_improve += 1
        if epoch_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

print("Finished Training")