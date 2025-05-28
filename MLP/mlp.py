import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# 设置设备：GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# 1. 加载 .mat 格式的数据
# ==============================
file_path = r'E:\my_project\over\Intelligent_Analysis\data\MINIST_final1.mat'
mat_data = sio.loadmat(file_path)

train_data = mat_data['train_data'].astype(np.float32)     # shape: (3000, 784)
train_labels = mat_data['train_labels'].ravel().astype(int)  # shape: (3000,)
test_data = mat_data['test_data'].astype(np.float32)       # shape: (1000, 784)
test_labels = mat_data['test_labels'].ravel().astype(int)    # shape: (1000,)

print("数据加载完成...")

# ==============================
# 2. 数据预处理 + 转换为 Tensor
# ==============================
# 归一化到 [0, 1]
train_data = train_data / 255.0
test_data = test_data / 255.0

# 转为 PyTorch Tensor
train_tensor_data = torch.tensor(train_data)
train_tensor_labels = torch.tensor(train_labels)
test_tensor_data = torch.tensor(test_data)
test_tensor_labels = torch.tensor(test_labels)

# 构建 Dataset 和 DataLoader
train_dataset = TensorDataset(train_tensor_data, train_tensor_labels)
test_dataset = TensorDataset(test_tensor_data, test_tensor_labels)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

print("数据预处理与加载器构建完成...")

# ==============================
# 3. 定义4层全连接神经网络 + Kaiming 初始化
# ==============================
class MLP_4Layer(nn.Module):
    def __init__(self):
        super(MLP_4Layer, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

model = MLP_4Layer().to(device)

# ==============================
# 4. 损失函数和优化器
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# 5. 训练模型
# ==============================
def train(model, epochs=100):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# ==============================
# 6. 测试模型
# ==============================
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

# ==============================
# 7. 运行训练和测试
# ==============================
train(model, epochs=100)
test(model)

# ==============================
# 8. 可视化预测结果
# ==============================
def visualize_predictions(model, num_images=6):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 3, i + 1)
        img = images[i].cpu().numpy().reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(model)


# ==============================
# 9. 绘制混淆矩阵
# ==============================

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# 自定义绘图函数（你已写好）
def plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

# ==============================
# 10. 获取模型预测结果并绘制混淆矩阵
# ==============================
def evaluate_and_plot_confusion_matrix(model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 类别名称（数字 0~9）
    class_names = [str(i) for i in range(10)]

    # 调用自定义绘图函数
    plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix - MLP Model')


evaluate_and_plot_confusion_matrix(model)