from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Subset
criterion = nn.CrossEntropyLoss()
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tcm_image.datasets import ImageClassificationDataset
from tcm_image.datasets.torch import ImageDataset
import torchvision.transforms as transforms
import pytest
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns


# 参数配置
base="resnet50"
category='coatingcolor'
lr=0.00001
pkl_root='C:/Users/LENOVO/PycharmProjects/tcm-image-dataset/output/20241118-213353/singlelabel_CoatingColor.pkl'


num_epochs=50
batch_size=128

# 数据加载
dataset = ImageClassificationDataset.load(pkl_root, mode='pickle')

# 分割数据集
train_dataset, test_dataset = dataset.split(0.8,True)
print(f'Train dataset size:{len(train_dataset)},Test dataset size:{len(test_dataset)}')

# 数据预处理
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪图像到224x224像素的大小
        transforms.RandomHorizontalFlip(p=0.5),  # 一定的概率（默认为0.5）随机水平翻转图像
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 将 PIL Image 或 NumPy ndarray 转换为 FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# # 转换为torch数据集
train_dataset=ImageDataset.from_dataset_obj(train_dataset,transforms=train_transform)
test_dataset=ImageDataset.from_dataset_obj(test_dataset,transforms=test_transform)
#创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)
#遍历数据加载器
for batch in train_loader:
    images, labels = batch
    print(images.shape, labels.shape)
    break


# 加载预训练的ResNet-50模型
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# 只修改模型的第一层
new_conv1 = nn.Conv2d(640, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.conv1 = new_conv1
num_class = 2
# 修改最后一层以适应数据集的类别数
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_class)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=0.000001)


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 训练模型

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # 定义学习率调度器（如果需要）

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for inputs, labels in train_loader:
            inputs = inputs.float()  # 确保输入数据类型为 float
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备

            # 后向传播与优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸问题
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        train_losses.append(running_loss/len(train_dataset))
        train_accuracies.append(100. * corrects / total)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, '
              f'Accuracy: {train_accuracies[-1]:.2f}%')

        scheduler.step()  # 更新学习率
        test_model(model, test_loader, criterion)
    return(train_losses,train_accuracies)





def test_model(model, test_loader, criterion):
    # 测试代码
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total=0
    with torch.no_grad():
        # 前向传播
        for inputs, labels in test_loader:
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item()

            preds = output.argmax(dim=1)
            correct += preds.eq(labels.view_as(preds)).sum().item()
            total+=labels.size(0)
    test_losses.append(test_loss/len(test_dataset))
    test_accuracies.append(correct/total)
    print(f'Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%')
    return (test_losses, test_accuracies)




def evaluate_metrics(model,test_loader):
    model.eval()  # 设置模型为评估模式
# 初始化列表以收集所有标签和预测
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float)
            inputs, tagets = inputs.to(device), labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 获取预测的类别索引
            preds = outputs.argmax(dim=1)
            # 收集预测结果和真实标签
            all_labels.extend(tagets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
    # 计算性能指标
    cm = confusion_matrix(all_labels,all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # 打印性能指标
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    # 使用seaborn绘制混淆矩阵的热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    # 将指标保存到字典中
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    return metrics




import pandas as pd

def save_to_excel(train_losses, train_accuracies, test_losses, test_accuracies, metrics, num_epochs):
    epochs = range(1, num_epochs + 1)

    results = {
        'Epoch': epochs,
        'Train Loss': train_losses,  # 截取到 num_epochs
        'Train Accuracy': train_accuracies,
        'Test Loss': test_losses,
        'Test Accuracy': test_accuracies
    }

    results_df = pd.DataFrame(results)

    # 保存为 Excel 文件
    excel_path =  r'd:\Users\LENOVO\Desktop\CoatingColor.results.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        results_df.to_excel(writer, sheet_name='Training and Testing Results', index=False)


        # 将评估指标保存到Excel文件的第二个工作表
        metrics_df = pd.DataFrame(metrics, index=[0])  # 将字典转换为单行DataFrame
        metrics_df.to_excel(writer, sheet_name='Evaluation Metrics', index=False)
        print('Training, testing results and evaluation metrics saved to results.xlsx')

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs)

# 评估模型并保存指标
metrics = evaluate_metrics(model, test_loader)

# 保存训练和测试结果到Excel文件
save_to_excel(train_losses, train_accuracies, test_losses, test_accuracies, metrics, num_epochs)

# 保存模型
torch.save(model.state_dict(), 'resnet34.pth')

# 加载模型
model.load_state_dict(torch.load('resnet34.pth'))

# 设置为评估模式
model.eval()

# 可视化训练和测试的Loss和Accuracy
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 绘制Loss曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs[:len(test_losses)], test_losses, label='Test Loss')  # 避免test_losses比train_losses短
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs[:len(test_accuracies)], test_accuracies, label='Test Accuracy')  # 同样避免不匹配
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)


