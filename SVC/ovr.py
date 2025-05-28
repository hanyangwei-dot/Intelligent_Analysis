from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import SVC as CL  # 使用标准的SVC类


class ovr:
    def __init__(self, **svc_params):
        """
        初始化 One-vs-Rest 分类器。

        参数:
        - svc_params: 传递给 SVC 的参数
        """
        self.svc_params = svc_params
        self.estimators_ = {}
        self.classes_ = None

    def fit(self, X, y):
        """
        训练 One-vs-Rest 分类器。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 标签向量 (n_samples,)
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        for i, class_label in enumerate(self.classes_):
            # 创建针对每个类别的二分类器
            mask = (y == class_label)
            y_binary = np.where(mask, 1, -1)

            estimator = CL.SVC(**self.svc_params)
            estimator.fit(X, y_binary)
            print(f'{class_label} vs Rest分类器训练完成...')  # debug
            self.estimators_[class_label] = estimator

        return self

    def predict(self, X):
        """
        使用训练好的 One-vs-Rest 分类器进行预测。

        参数:
        - X: 特征矩阵 (n_samples, n_features)

        返回:
        - 预测的类别标签 (n_samples,)
        """
        X = check_array(X)
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_))

        # 并行预测
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(
                delayed(self._predict_single_classifier)(X, class_label)
                for class_label in self.estimators_
            )

        # 收集预测结果
        for class_label, predictions in results:
            scores[:, self.classes_.tolist().index(class_label)] = predictions

        # 选择获得最高分数的类别
        predicted_classes = self.classes_[np.argmax(scores, axis=1)]
        return predicted_classes

    # def _predict_single_classifier(self, X, class_label):
    #     """
    #     对单个二分类器进行预测。
    #
    #     参数:
    #     - X: 特征矩阵 (n_samples, n_features)
    #     - class_label: 类别标签
    #
    #     返回:
    #     - 类别标签和预测结果
    #     """
    #     estimator = self.estimators_[class_label]
    #     predictions = estimator.decision_function(X) if hasattr(estimator,
    #                                                             "decision_function") else estimator.predict_proba(X)[:,
    #                                                                                       1]
    #     return class_label, predictions
    def _predict_single_classifier(self, X, class_label):
        """
        对单个二分类器进行预测。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - class_label: 类别标签

        返回:
        - 类别标签和预测结果（决策函数的输出）
        """
        estimator = self.estimators_[class_label]


        # if hasattr(estimator, "decision_function"):
        predictions = estimator.decisionFunction(X)
        # else:
        #     # 如果分类器不支持decision_function，则抛出异常
        #     raise AttributeError(f"Estimator for class {class_label} does not support 'decisionFunction'. "
        #                          f"Consider using a different classifier that supports it.")

        return class_label, predictions

    def score(self, X, y):
        """
        计算模型的准确率。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 真实标签 (n_samples,)

        返回:
        - 模型的准确率
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def cross_val_score(self, X, y, cv=5):
        """
        执行K折交叉验证并返回平均准确率及标准差。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 标签向量 (n_samples,)
        - cv: 折叠数量，默认为5

        返回:
        - 平均准确率和标准差
        """
        X, y = check_X_y(X, y)
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)  # 设置随机种子保证可重复性
        scores = []

        for train_index, val_index in kfold.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # 训练模型
            self.fit(X_train, y_train)

            # 评估模型
            score = self.score(X_val, y_val)
            scores.append(score)

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        print(f"Cross-validation accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        return mean_accuracy, std_accuracy

    def evaluate_single_classifier(self, X, y, class_label):
        """
        评估单个二分类器的性能。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 标签向量 (n_samples,)
        - class_label: 要评估的类别标签

        返回:
        - 该二分类器的准确率
        """
        if class_label not in self.estimators_:
            raise ValueError(f"Class label {class_label} does not exist in the trained classifiers.")

        # 获取对应的二分类器
        estimator = self.estimators_[class_label]

        # 进行预测
        y_binary = np.where(y == class_label, 1, -1)
        y_pred = estimator.predict(X)

        # 计算准确率
        accuracy = accuracy_score(y_binary, y_pred)
        return accuracy

    def evaluate_all_classifiers(self, X, y):
        """
        评估所有二分类器的性能，并打印每个二分类器的准确率。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 标签向量 (n_samples,)
        """
        accuracies = {}
        for class_label in self.estimators_:
            accuracy = self.evaluate_single_classifier(X, y, class_label)
            accuracies[class_label] = accuracy
            print(f"Accuracy of classifier for class {class_label}: {accuracy:.4f}")

        return accuracies