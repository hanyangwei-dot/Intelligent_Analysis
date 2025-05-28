import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score

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
# 2. 数据预处理：归一化到 [0, 1]
# ==============================
def normalize(X):
    return X / 255.0

train_data_norm = normalize(train_data)
test_data_norm = normalize(test_data)

# ==============================
# 3. 初始化 OvR 分类器
# ==============================
from ovr import ovr

# 线性核 OvR-SVM
ovr_linear1 = ovr(C=1.0, kernel='linear')
ovr_linear2 = ovr(C=100.0, kernel='linear')

# RBF 核 OvR-SVM
ovr_rbf1 = ovr(C=1.0, kernel='rbf', γ=0.1)
ovr_rbf2 = ovr(C=100.0, kernel='rbf', γ=0.05)

# ==============================
# 4. 训练模型
# ==============================
print("开始训练线性核 OvR SVM1...")
ovr_linear1.fit(train_data_norm, train_labels)

print("开始训练线性核 OvR SVM2...")
ovr_linear2.fit(train_data_norm, train_labels)

print("开始训练 RBF 核 OvR SVM1...")
ovr_rbf1.fit(train_data_norm, train_labels)

print("开始训练 RBF 核 OvR SVM2...")
ovr_rbf2.fit(train_data_norm, train_labels)

# ==============================
# 5. 测试集预测与评估
# ==============================
print("在测试集上进行预测...")

y_pred_linear1 = ovr_linear1.predict(test_data_norm)
y_pred_linear2 = ovr_linear2.predict(test_data_norm)
y_pred_rbf1 = ovr_rbf1.predict(test_data_norm)
y_pred_rbf2 = ovr_rbf2.predict(test_data_norm)

acc_linear1 = accuracy_score(test_labels, y_pred_linear1)
acc_linear2 = accuracy_score(test_labels, y_pred_linear2)
acc_rbf1 = accuracy_score(test_labels, y_pred_rbf1)
acc_rbf2 = accuracy_score(test_labels, y_pred_rbf2)

# ==============================
# 6. 输出结果
# ==============================
print(f"\n【线性核 OvR SVM1 测试准确率】: {acc_linear1 * 100:.2f}%")
print(f"\n【线性核 OvR SVM2 测试准确率】: {acc_linear2 * 100:.2f}%")
print(f"\n【RBF 核 OvR SVM1 测试准确率】: {acc_rbf1 * 100:.2f}%")
print(f"\n【RBF 核 OvR SVM2 测试准确率】: {acc_rbf2 * 100:.2f}%")
