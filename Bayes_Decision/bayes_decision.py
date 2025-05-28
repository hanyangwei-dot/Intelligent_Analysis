import numpy as np
import scipy.io as sio
from collections import defaultdict
import math
import time

# ----------------------------
# 1. 加载数据
# ----------------------------
file_path = r'E:\my_project\over\Intelligent_Analysis\data\MINIST_final1.mat'
mat_data = sio.loadmat(file_path)

train_data = mat_data['train_data'].astype(np.float64)     # shape: (3000, 784)
train_labels = mat_data['train_labels'].ravel()            # shape: (3000,)
test_data = mat_data['test_data'].astype(np.float64)       # shape: (1000, 784)
test_labels = mat_data['test_labels'].ravel()              # shape: (1000,)

print("数据加载完成...")

# ----------------------------
# 2. 数据归一化（将像素值从 [0, 255] 缩放到 [0, 1]）
# ----------------------------
train_data /= 255.0
test_data /= 255.0

print("数据已归一化到 [0, 1] 范围...")

# ----------------------------
# 3. 构建模型参数：均值、方差、先验
# ----------------------------
class_stats = dict()
classes = np.unique(train_labels)

start_time = time.time()

for cls in classes:
    idxs = (train_labels == cls)
    X_cls = train_data[idxs]

    mean = X_cls.mean(axis=0)               # (784,)
    std = X_cls.std(axis=0)                 # (784,)
    prior = X_cls.shape[0] / train_data.shape[0]

    class_stats[cls] = {
        'mean': mean,
        'std': std,
        'prior': prior
    }

end_time = time.time()
print(f"训练统计量计算完成，耗时 {end_time - start_time:.2f} 秒")

# ----------------------------
# 4. 高斯概率密度函数（取 log 避免下溢）
# ----------------------------
def log_gaussian_prob(x, mu, sigma):
    # 防止除以零
    eps = 1e-6
    sigma += eps
    return -0.5 * np.log(2 * math.pi * sigma ** 2) - ((x - mu) ** 2) / (2 * sigma ** 2)

# ----------------------------
# 5. 分类预测函数
# ----------------------------
def predict(x):
    best_class = None
    best_score = -np.inf

    for cls, stats in class_stats.items():
        log_prior = np.log(stats['prior'])
        log_likelihood = log_gaussian_prob(x, stats['mean'], stats['std']).sum()
        score = log_prior + log_likelihood

        if score > best_score:
            best_score = score
            best_class = cls

    return best_class

# ----------------------------
# 6. 测试集预测
# ----------------------------
correct = 0
y_pred = []

start_time = time.time()

for i in range(test_data.shape[0]):
    x = test_data[i]
    y_true = test_labels[i]
    y_hat = predict(x)
    y_pred.append(y_hat)

    if y_hat == y_true:
        correct += 1

end_time = time.time()
print(f"预测完成，耗时 {end_time - start_time:.2f} 秒")

# ----------------------------
# 7. 准确率评估
# ----------------------------
accuracy = correct / test_data.shape[0]
print(f"\n【测试集准确率】: {accuracy * 100:.2f}%")


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 8. 构建混淆矩阵
# ----------------------------
num_classes = len(classes)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, pred_label in zip(test_labels, y_pred):
    conf_matrix[int(true_label), int(pred_label)] += 1

print("\n【混淆矩阵】:")
print(conf_matrix)

# ----------------------------
# 9. 可视化混淆矩阵
# ----------------------------
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# 在矩阵中添加数字标签
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
