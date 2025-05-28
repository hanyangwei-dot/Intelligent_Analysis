import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# import seaborn as sns
from collections import defaultdict, Counter

# ==============================
# 自定义 KMeans 类
# ==============================
class KMeans:
    def __init__(self, k=10, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 随机初始化中心点
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        for _ in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros((self.k, n_features))
            counts = np.zeros(self.k)

            for i in range(n_samples):
                cluster_idx = labels[i]
                new_centroids[cluster_idx] += X[i]
                counts[cluster_idx] += 1

            for j in range(self.k):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            diff = X - self.centroids[i]
            distances[:, i] = np.linalg.norm(diff, axis=1)
        return distances


# ==============================
# 数据加载与预处理
# ==============================
file_path = r'E:\my_project\over\Intelligent_Analysis\data\MINIST_final1.mat'
mat_data = sio.loadmat(file_path)

train_data = mat_data['train_data'].astype(np.float64)     # shape: (3000, 784)
train_labels = mat_data['train_labels'].ravel().astype(int)  # shape: (3000,)
test_data = mat_data['test_data'].astype(np.float64)       # shape: (1000, 784)
test_labels = mat_data['test_labels'].ravel().astype(int)    # shape: (1000,)

print("数据加载完成...")

def normalize(X):
    return X / 255.0

train_data_norm = normalize(train_data)
test_data_norm = normalize(test_data)


# ==============================
# 模型训练与预测
# ==============================
print("开始训练 K-Means...")
kmeans = KMeans(k=10, max_iters=50)
kmeans.fit(train_data_norm)

# 对测试集进行聚类
test_cluster_labels = kmeans.predict(test_data_norm)


# ==============================
# 映射聚类标签到实际类别
# ==============================
def map_clusters_to_labels(clusters, true_labels, k=10):
    mapping = {}
    for cluster_id in range(k):
        idxs = (clusters == cluster_id)
        labels_in_cluster = true_labels[idxs]
        if len(labels_in_cluster) == 0:
            mapping[cluster_id] = -1  # 如果簇为空，标记为 -1
        else:
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            mapping[cluster_id] = most_common_label

    mapped_labels = np.array([mapping[c] for c in clusters])
    return mapped_labels

mapped_test_labels = map_clusters_to_labels(test_cluster_labels, test_labels)


# ==============================
# 准确率评估
# ==============================
accuracy = np.mean(mapped_test_labels == test_labels)
print(f"\n【测试集近似准确率】: {accuracy * 100:.2f}%")


# ==============================
# 构建混淆矩阵
# ==============================
def build_confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix

conf_matrix = build_confusion_matrix(test_labels, mapped_test_labels)
print("\n【混淆矩阵】:\n", conf_matrix)


# ==============================
# 绘制混淆矩阵
# ==============================
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

# 类别名称，这里直接使用数字 0-9 表示类别
class_names = [str(i) for i in range(10)]

# 调用绘图函数
plot_confusion_matrix(conf_matrix, class_names)

