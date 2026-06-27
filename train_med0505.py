import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 硬體與參數設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'medproject/dataset/output_dataset0626' 
batch_size = 32        
num_epochs = 100       
learning_rate = 0.0001 

# --- 2. 數據增強 (360度旋轉 + 銳利化) ---
data_transforms = {
    'train': transforms.Compose([

        transforms.Resize((448, 448)),

        # 藥品角度變化
        transforms.ColorJitter(
        brightness=0.4,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
        ),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        # 光照、色偏
        transforms.ColorJitter(
            brightness=0.6,
            contrast=0.4,
            saturation=0.4,
            hue=0.05
        ),

        # 銳利化
        transforms.RandomAdjustSharpness(
            sharpness_factor=2,
            p=0.3
        ),

        # 模擬失焦
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.5)
        ),

        transforms.ToTensor(),

        # 模擬遮擋
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3)
        ),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

    ]),

    'val': transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}

# --- 3. 載入資料與自動平衡採樣 ---
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# 💡 建立 WeightedRandomSampler (解決 Surin 只有 10 張的問題)
train_labels = image_datasets['train'].targets
class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
weight = 1. / class_sample_count
samples_weight = torch.from_numpy(weight[train_labels])
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

# 注意：使用 sampler 時，shuffle 必須為 False
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
}

class_names = image_datasets['train'].classes
print(f"✅ 成功載入 {len(class_names)} 個類別")

# --- 4. 建立模型 (ResNet50) ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names)) 
model = model.to(device)

# --- 5. 損失函數與優化器 (精準權重分配) ---
weights = torch.ones(len(class_names)).to(device)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --- 6. 訓練函式 ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss, running_corrects = 0.0, 0
            # 只在 validation 收集結果
            all_labels = []
            all_preds = []
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'val':
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train': scheduler.step()
            
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_resnet50_448px_0627.pth')
                print(f"🌟 發現更佳模型，已儲存！ Acc: {best_acc:.4f}")

                # ========= Confusion Matrix =========
                cm = confusion_matrix(
                    all_labels,
                    all_preds,
                    normalize='true'
                )

                plt.figure(figsize=(14,12))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names
                )

                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Confusion Matrix Epoch {epoch+1}")

                plt.tight_layout()

                plt.savefig("best_confusion_matrix.png")
                plt.close()

                print("✅ Confusion Matrix 已儲存")

                print("\n================ Classification Report ================\n")

                report = classification_report(
                    all_labels,
                    all_preds,
                    target_names=class_names,
                    digits=4
                )

                print(report)

                with open("best_classification_report.txt", "w", encoding="utf-8") as f:
                    f.write(report)
    
    return model

if __name__ == '__main__':
    train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)