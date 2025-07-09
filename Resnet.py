import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
%matplotlib
inline
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

from torchvision import transforms
import random


# 自定义伽马校正
class GammaCorrection:
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        return transforms.functional.adjust_gamma(img, gamma)


# 自定义亮度调整
class BrightnessAdjust:
    def __init__(self, brightness_range=(-0.1, 0.1)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        brightness = random.uniform(*self.brightness_range)
        return transforms.functional.adjust_brightness(img, 1 + brightness)


# 训练集图像预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    BrightnessAdjust(),
    GammaCorrection(gamma_range=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试集图像预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径
dataset_dir = 'RDE_split'
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')
print('训练集路径', train_path)
print('测试集路径', test_path)

from torchvision import datasets

# 载入数据集
train_dataset = datasets.ImageFolder(train_path, train_transform)
test_dataset = datasets.ImageFolder(test_path, test_transform)

print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)
print('测试集图像数量', len(test_dataset))

# 类别映射
class_names = train_dataset.classes
n_class = len(class_names)
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}

# 保存映射关系
os.makedirs('checkpoint', exist_ok=True)
np.save('checkpoint/idx_to_labels_resnet18.npy', idx_to_labels)
np.save('checkpoint/labels_to_idx_resnet18.npy', train_dataset.class_to_idx)

from torch.utils.data import DataLoader

BATCH_SIZE = 32

# 数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

from torchvision import models
import torch.optim as optim

# 创建模型
model = models.resnet18(pretrained=True)

# 冻结前10个卷积层
ct = 0
for child in model.children():
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False
    ct += 1

# 修改全连接层
model.fc = nn.Linear(model.fc.in_features, n_class)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 论文参数

# 学习率调度器
lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
    verbose=True
)


def train_one_batch(images, labels):
    images = images.to(device)
    labels = labels.to(device)

    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取预测结果
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # 计算评估指标（添加召回率和F1分数）
    log_train = {
        'epoch': epoch,
        'batch': batch_idx,
        'train_loss': loss,
        'train_accuracy': accuracy_score(labels, preds),
        'train_recall': recall_score(labels, preds, average='macro'),
        'train_f1_score': f1_score(labels, preds, average='macro')
    }
    return log_train


def evaluate_testset():
    model.eval()
    loss_list = []
    labels_list = []
    preds_list = []
    probas_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss_list.append(loss.item())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probas_list.extend(F.softmax(outputs, dim=1).cpu().numpy())

    # 计算多种评估指标
    log_test = {
        'epoch': epoch,
        'test_loss': np.mean(loss_list),
        'test_accuracy': accuracy_score(labels_list, preds_list),
        'test_precision': precision_score(labels_list, preds_list, average='macro'),
        'test_recall': recall_score(labels_list, preds_list, average='macro'),
        'test_f1_score': f1_score(labels_list, preds_list, average='macro')
    }
    return log_test


# 初始化变量
epoch = 0
batch_idx = 0
best_test_accuracy = 0
best_f1_score = 0
df_train_log = pd.DataFrame()
df_test_log = pd.DataFrame()

# 训练循环
EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}/{EPOCHS}')
    model.train()

    # 训练阶段
    epoch_train_loss = []
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}'):
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        epoch_train_loss.append(log_train['train_loss'])
        df_train_log = pd.concat([df_train_log, pd.DataFrame([log_train])], ignore_index=True)

    # 更新学习率
    avg_train_loss = np.mean(epoch_train_loss)
    lr_scheduler.step(avg_train_loss)

    # 测试阶段
    log_test = evaluate_testset()
    df_test_log = pd.concat([df_test_log, pd.DataFrame([log_test])], ignore_index=True)

    # 保存最佳模型
    current_f1 = log_test['test_f1_score']
    if current_f1 > best_f1_score:
        best_f1_score = current_f1
        torch.save(model.state_dict(), f'checkpoint/best_resnet18_epoch{epoch}_f1{best_f1_score:.3f}.pth')
        print(f'保存新的最佳模型: F1分数 {best_f1_score:.3f}')

# 保存训练日志
df_train_log.to_csv('training_log_train.csv', index=False)
df_test_log.to_csv('training_log_test.csv', index=False)

# 载入最佳模型
best_model = models.resnet18(pretrained=False)
best_model.fc = nn.Linear(best_model.fc.in_features, n_class)
best_model.load_state_dict(torch.load(f'checkpoint/best_resnet18_epoch{epoch}_f1{best_f1_score:.3f}.pth'))
best_model = best_model.to(device)
best_model.eval()

# 最终评估
final_log = evaluate_testset()
print("\n最终测试结果:")
print(f"测试准确率: {final_log['test_accuracy']:.4f}")
print(f"测试召回率: {final_log['test_recall']:.4f}")
print(f"测试F1分数: {final_log['test_f1_score']:.4f}")

# 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_train_log.groupby('epoch')['train_loss'].mean(), label='训练损失')
plt.plot(df_test_log['test_loss'], label='测试损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df_train_log.groupby('epoch')['train_accuracy'].mean(), label='训练准确率')
plt.plot(df_test_log['test_accuracy'], label='测试准确率')
plt.title('准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()