import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. 加载 MNIST 数据
# ==============================
file_path = r'E:\my_project\over\Intelligent_Analysis\data\MINIST_final1.mat'
mat_data = sio.loadmat(file_path)

train_data = mat_data['train_data'].astype(np.float64)     # shape: (3000, 784)
print(train_data.shape)
train_labels = mat_data['train_labels'].ravel().astype(int)  # shape: (3000,)
test_data = mat_data['test_data'].astype(np.float64)       # shape: (1000, 784)
print(test_data.shape)
test_labels = mat_data['test_labels'].ravel().astype(int)    # shape: (1000,)

print("数据加载完成...")

# ==============================
# 2. 数据预处理：标准化
# ==============================
def normalize(X):
    return X / 255.0

train_data_norm = normalize(train_data)
test_data_norm = normalize(test_data)

# ==============================
# 3. 初始化 SVM 模型（OvO 多分类）
# ==============================

# 线性核 SVM（默认使用 OvO 多分类）
svm_linear1 = SVC(C=1.0, kernel='linear', decision_function_shape='ovr')
svm_linear2 = SVC(C=100.0, kernel='linear', decision_function_shape='ovr')

# RBF 核 SVM
svm_rbf1 = SVC(C=1.0, kernel='rbf', gamma=0.1, decision_function_shape='ovr')
svm_rbf2 = SVC(C=100.0, kernel='rbf', gamma=0.05, decision_function_shape='ovr')

# ==============================
# 4. 训练模型
# ==============================
print("开始训练线性核 SVM1...")
svm_linear1.fit(train_data_norm, train_labels)

print("开始训练线性核 SVM2...")
svm_linear2.fit(train_data_norm, train_labels)

print("开始训练 RBF 核 SVM1...")
svm_rbf1.fit(train_data_norm, train_labels)

print("开始训练 RBF 核 SVM2...")
svm_rbf2.fit(train_data_norm, train_labels)

# ==============================
# 5. 测试集预测与评估
# ==============================
print("在测试集上进行预测...")

y_pred_linear1 = svm_linear1.predict(test_data_norm)
y_pred_linear2 = svm_linear2.predict(test_data_norm)
y_pred_rbf1 = svm_rbf1.predict(test_data_norm)
y_pred_rbf2 = svm_rbf2.predict(test_data_norm)

acc_linear1 = accuracy_score(test_labels, y_pred_linear1)
acc_linear2 = accuracy_score(test_labels, y_pred_linear2)
acc_rbf1 = accuracy_score(test_labels, y_pred_rbf1)
acc_rbf2 = accuracy_score(test_labels, y_pred_rbf2)

# ==============================
# 6. 输出结果
# ==============================
print(f"\n【线性核 SVM1 测试准确率】: {acc_linear1 * 100:.2f}%")
print(f"\n【线性核 SVM2 测试准确率】: {acc_linear2 * 100:.2f}%")
print(f"\n【RBF 核 SVM1 测试准确率】: {acc_rbf1 * 100:.2f}%")
print(f"\n【RBF 核 SVM2 测试准确率】: {acc_rbf2 * 100:.2f}%")

# ==============================
# 7. 绘制混淆矩阵svm_rbf2 = SVC(C=100.0, kernel='rbf', gamma=0.05, decision_function_shape='ovr')
# ==============================
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# 自定义绘图函数
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
# 8. 绘制 svm_rbf2 的混淆矩阵
# ==============================
print("绘制 SVM (RBF 核) 的混淆矩阵...")

# 获取预测结果（确保你已经运行过预测）
y_true = test_labels
y_pred = y_pred_rbf2  # 这是之前 svm_rbf2.predict(test_data_norm) 的结果

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)

# 类别名称（数字 0~9）
class_names = [str(i) for i in range(10)]

# 调用绘图函数
plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix - RBF SVM (C=100, gamma=0.05)')