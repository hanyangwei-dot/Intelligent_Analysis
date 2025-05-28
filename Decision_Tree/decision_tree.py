import numpy as np
import scipy.io as sio
import math
from collections import Counter

# ----------------------------
# 1. 加载数据
# ----------------------------
file_path = r'E:\my_project\over\Intelligent_Analysis\data\MINIST_final1.mat'
mat_data = sio.loadmat(file_path)

train_data = mat_data['train_data'].astype(np.float64)     # shape: (3000, 784)
train_labels = mat_data['train_labels'].ravel().astype(int)  # shape: (3000,)
test_data = mat_data['test_data'].astype(np.float64)       # shape: (1000, 784)
test_labels = mat_data['test_labels'].ravel().astype(int)    # shape: (1000,)

print("数据加载完成...")

# ----------------------------
# 数据归一化到 [0, 1]
# ----------------------------
train_data /= 255.0
test_data /= 255.0

print("数据已归一化到 [0, 1] 范围...")


# ----------------------------
# 2. 特征离散化：将 [0, 1] 转换为 0 或 1（二值化）
# ----------------------------
def binarize(X, threshold=0.1):
    return (X > threshold).astype(int)

train_data_bin = binarize(train_data)
test_data_bin = binarize(test_data)

print("特征已二值化...")

# ----------------------------
# 3. 信息熵计算函数
# ----------------------------
def entropy(y):
    counts = np.bincount(y)
    ps = counts / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# ----------------------------
# 4. 条件熵与信息增益
# ----------------------------
def conditional_entropy(X_col, y):
    unique_vals = np.unique(X_col)
    total_entropy = 0
    for val in unique_vals:
        idxs = X_col == val
        sub_y = y[idxs]
        weight = len(sub_y) / len(y)
        total_entropy += weight * entropy(sub_y)
    return total_entropy

def info_gain(X_col, y):
    return entropy(y) - conditional_entropy(X_col, y)

# ----------------------------
# 5. 决策树节点类（字典结构）
# ----------------------------
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 当前节点使用的特征索引
        self.threshold = threshold  # 阈值（本例中固定为 0.5，已离散化）
        self.left = left            # 左子树（特征 <= threshold）
        self.right = right          # 右子树（特征 > threshold）
        self.value = value          # 如果是叶节点，保存预测类别

# ----------------------------
# 6. 构建决策树（递归）
# ----------------------------
def build_tree(X, y, depth=0, max_depth=5):
    # 终止条件：纯度高 or 深度限制
    if len(set(y)) == 1 or depth >= max_depth:
        leaf_value = Counter(y).most_common(1)[0][0]
        return DecisionTreeNode(value=leaf_value)

    num_features = X.shape[1]
    best_gain = -1
    best_feature = None

    # 遍历所有特征，找信息增益最大的特征
    for feature_idx in range(num_features):
        gain = info_gain(X[:, feature_idx], y)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_idx

    # 如果没有增益，返回多数类
    if best_gain == 0:
        leaf_value = Counter(y).most_common(1)[0][0]
        return DecisionTreeNode(value=leaf_value)

    # 划分数据集
    left_idxs = X[:, best_feature] == 0
    right_idxs = X[:, best_feature] == 1

    left_subtree = build_tree(X[left_idxs], y[left_idxs], depth + 1, max_depth)
    right_subtree = build_tree(X[right_idxs], y[right_idxs], depth + 1, max_depth)

    return DecisionTreeNode(feature=best_feature, left=left_subtree, right=right_subtree)

# ----------------------------
# 7. 预测单个样本
# ----------------------------
def predict_sample(tree, x):
    if tree.value is not None:
        return tree.value
    feature_val = x[tree.feature]
    if feature_val == 0:
        return predict_sample(tree.left, x)
    else:
        return predict_sample(tree.right, x)

# ----------------------------
# 8. 主程序：训练 & 测试
# ----------------------------
print("开始训练决策树...")
tree = build_tree(train_data_bin, train_labels, max_depth=5)
print("决策树训练完成...")

# 预测测试集
correct = 0
for i in range(test_data_bin.shape[0]):
    pred = predict_sample(tree, test_data_bin[i])
    if pred == test_labels[i]:
        correct += 1

accuracy = correct / len(test_labels)
print(f"\n【测试集准确率】: {accuracy * 100:.2f}%")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# ----------------------------
# 9. 构建混淆矩阵
# ----------------------------
def build_confusion_matrix(y_true, y_pred, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[true_label][pred_label] += 1
    return conf_matrix

# 预测整个测试集（为了后续绘制混淆矩阵）
y_true = test_labels
y_pred = [predict_sample(tree, sample) for sample in test_data_bin]

num_classes = 10  # 手写数字为 0~9，共 10 类
conf_matrix = build_confusion_matrix(y_true, y_pred, num_classes)

print("\n【混淆矩阵】:")
print(conf_matrix)

# ----------------------------
# 10. 可视化混淆矩阵
# ----------------------------
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

    # 在矩阵中显示数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

# 类别名称（数字 0~9）
class_names = [str(i) for i in range(10)]
plot_confusion_matrix(conf_matrix, class_names)