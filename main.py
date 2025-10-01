import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from alexnet_model import AlexNet
from data_loader import get_dataloaders
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 记录模型参数和梯度
def log_gradients_and_weights(model, writer, global_step):
    if global_step % 100 == 0:
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                writer.add_histogram(f'Weights/{name}', param, global_step)
                writer.add_histogram(f'Gradients/{name}', param.grad, global_step)

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    hparams = {
        'DATASET_PATH': './CatADog_5340',
        'BATCH_SIZE': 32,
        'NUM_CLASSES': 2,
        'NUM_EPOCHS': 50,
        'LEARNING_RATE': 0.001,
        'EARLY_STOPPING_PATIENCE': 5,
        'OPTIMIZER': 'Adam'
    }

    print("Loading data...")
    train_loader, test_loader = get_dataloaders(
        dataset_root_path=hparams['DATASET_PATH'],
        batch_size=hparams['BATCH_SIZE']
    )
    print("Data loaded successfully.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AlexNet(num_classes=hparams['NUM_CLASSES']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    epoch_no_improve = 0
    best_val_acc = 0.0
    
    print("Starting training...")
    writer = SummaryWriter('runs/alexnet_experiment_rich_logs')
    
    # 记录模型结构和一些样本图像
    # 从训练集中取一个批次的图像用来可视化模型图
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    writer.add_graph(model, images.to(device))

    # 记录一个批次的训练图像，检查数据增强效果
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('train_images_sample', img_grid)
    
    global_step = 0
    for epoch in range(hparams['NUM_EPOCHS']):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 记录梯度
            log_gradients_and_weights(model, writer, global_step)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 记录训练损失 
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1

        # 每个epoch的平均训练损失
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for val_inputs, val_labels in test_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                outputs = model(val_inputs)
                val_loss = criterion(outputs, val_labels)
                val_running_loss += val_loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
                
                # 收集所有预测和标签
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        # 记录更丰富的验证指标
        accuracy = 100 * correct / total
        val_epoch_loss = val_running_loss / len(test_loader)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        writer.add_scalar('Loss/validation', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)
        writer.add_scalar('Metrics/F1-Score', f1, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"Epoch [{epoch+1}/{hparams['NUM_EPOCHS']}] -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {accuracy:.2f}%")
        
        scheduler.step(accuracy)

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            epoch_no_improve = 0
            torch.save(model.state_dict(), 'best_alexnet_model_CatADog.pth')
            print(f"  -> Val Acc Improved to {best_val_acc:.2f}%. Saving model...")
        else:
            epoch_no_improve += 1

        if epoch_no_improve >= hparams['EARLY_STOPPING_PATIENCE']:
            print(f"Early stopping triggered after {hparams['EARLY_STOPPING_PATIENCE']} epochs without improvement.")
            break
    
    # 记录超参数和最终指标
    writer.add_hparams(
        {
            'lr': hparams['LEARNING_RATE'],
            'batch_size': hparams['BATCH_SIZE'],
            'optimizer': hparams['OPTIMIZER']
        },
        {
            'accuracy': best_val_acc,
            'final_val_loss': val_epoch_loss,
            'f1_score': f1
        }
    )
    
    writer.close()
    print("Finished Training")

if __name__ == '__main__':
    set_seed(2025) 
    main()