import numpy as np
import scipy.io as sio
from collections import Counter
import matplotlib.pyplot as plt


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
def build_tree(X, y, depth=0, max_depth=5, feature_indices=None):
    # 终止条件：纯度高 or 深度限制
    if len(set(y)) == 1 or depth >= max_depth:
        leaf_value = Counter(y).most_common(1)[0][0]
        return DecisionTreeNode(value=leaf_value)

    num_features = X.shape[1]
    best_gain = -1
    best_feature = None

    features_to_check = range(num_features) if feature_indices is None else range(len(feature_indices))

    # 遍历所有特征，找信息增益最大的特征
    for feature_idx in features_to_check:
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

    left_subtree = build_tree(X[left_idxs], y[left_idxs], depth + 1, max_depth, feature_indices)
    right_subtree = build_tree(X[right_idxs], y[right_idxs], depth + 1, max_depth, feature_indices)

    actual_feature = best_feature if feature_indices is None else feature_indices[best_feature]
    return DecisionTreeNode(feature=actual_feature, left=left_subtree, right=right_subtree)


# ----------------------------
# 7. 预测单个样本
# ----------------------------
def predict_sample(tree, x, is_forest=False):
    if tree.value is not None:
        return tree.value
    feature_val = x[tree.feature]
    if feature_val == 0:
        return predict_sample(tree.left, x, is_forest)
    else:
        return predict_sample(tree.right, x, is_forest)


# ----------------------------
# 8. 随机森林类（从零实现）
# ----------------------------
class RandomForest:
    def __init__(self, n_trees=50, max_depth=5, sample_ratio=0.5, feature_ratio=0.5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.trees = []

    def fit(self, X, y):
        """训练多棵决策树"""
        for _ in range(self.n_trees):
            tree = self._build_tree(X, y)
            self.trees.append(tree)

    def _build_tree(self, X, y):
        """构建单棵决策树（带 Bootstrap 和随机特征）"""
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Bootstrap 抽样（有放回）
        idxs = np.random.choice(n_samples, int(n_samples * self.sample_ratio), replace=True)
        X_sample, y_sample = X[idxs], y[idxs]

        # 随机选择部分特征
        n_selected_features = int(n_features * self.feature_ratio)
        selected_features = np.random.choice(n_features, n_selected_features, replace=False)

        return build_tree(X_sample[:, selected_features], y_sample, max_depth=self.max_depth,
                          feature_indices=selected_features)

    def predict(self, X):
        """对整个数据集进行预测，每棵树投票"""
        preds = []
        for x in X:
            tree_preds = [predict_sample(tree, x) for tree in self.trees]
            final_pred = Counter(tree_preds).most_common(1)[0][0]
            preds.append(final_pred)
        return preds


# ----------------------------
# 9. 构建混淆矩阵
# ----------------------------
def build_confusion_matrix(y_true, y_pred, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[true_label][pred_label] += 1
    return conf_matrix


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

    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()


# ----------------------------
# 11. 主程序：训练 & 测试（使用随机森林）
# ----------------------------
print("开始训练随机森林...")
rf = RandomForest(n_trees=10, max_depth=5, sample_ratio=0.8, feature_ratio=0.5)
rf.fit(train_data_bin, train_labels)
print("随机森林训练完成...")

# 预测测试集
y_pred = rf.predict(test_data_bin)
correct = sum(1 for true, pred in zip(test_labels, y_pred) if true == pred)
accuracy = correct / len(test_labels)
print(f"\n【测试集准确率】: {accuracy * 100:.2f}%")

# 构建并打印混淆矩阵
num_classes = 10
conf_matrix = build_confusion_matrix(test_labels, y_pred, num_classes)
print("\n【混淆矩阵】:")
print(conf_matrix)

# 可视化混淆矩阵
plot_confusion_matrix(conf_matrix, class_names=[str(i) for i in range(10)])