from itertools import combinations
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
import SVC as CL # 使用标准的SVC类
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class ovo:
    def __init__(self, **svc_params):
        """
        初始化 One-vs-One 分类器。

        参数:
        - svc_params: 传递给 SVC 的参数
        """
        self.svc_params = svc_params
        self.estimators_ = {}
        self.classes_ = None

    def fit(self, X, y):
        """
        训练 One-vs-One 分类器。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 标签向量 (n_samples,)
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        # print(unique_labels(y)) # debug
        self.n_classes_ = len(self.classes_)
        # print(len(self.classes_)) # debug

        # 创建所有类对的组合
        class_pairs = list(combinations(range(self.n_classes_), 2))
        # # 打印所有类对以进行调试
        # print("Generating all class pairs:")
        # for pair in class_pairs:
        #     print(f"  {pair}")

        # 创建所有类对的组合
        for i, j in class_pairs:
            class_pair = (i, j)
            # 获取属于这两个类别的样本
            mask = np.isin(y, [self.classes_[i], self.classes_[j]])
            X_pair = X[mask]
            y_pair = y[mask]

            # 将类别标签转换为 -1 和 1
            y_binary = np.where(y_pair == self.classes_[i], -1, 1)

            # 实例化并训练二分类器
            estimator = CL.SVC(**self.svc_params)
            estimator.fit(X_pair, y_binary)
            print(f'{class_pair}分类器训练完成...')  # debug
            self.estimators_[class_pair] = estimator

        return self

    def predict(self, X):
        """
        使用训练好的 One-vs-One 分类器进行预测。

        参数:
        - X: 特征矩阵 (n_samples, n_features)

        返回:
        - 预测的类别标签 (n_samples,)
        """
        X = check_array(X)
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))

        # 并行预测
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(
                delayed(self._predict_single_classifier)(X, class_pair)
                for class_pair in self.estimators_
            )

        # 收集投票结果
        for class_pair, predictions in results:
            for i, pred in enumerate(predictions):
                if pred == -1:
                    votes[i, class_pair[0]] += 1
                else:
                    votes[i, class_pair[1]] += 1

        # 选择获得最多票数的类别
        predicted_classes = self.classes_[np.argmax(votes, axis=1)]
        return predicted_classes

    def _predict_single_classifier(self, X, class_pair):
        """
        对单个二分类器进行预测。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - class_pair: 类对 (i, j)

        返回:
        - 类对和预测结果
        """
        estimator = self.estimators_[class_pair]
        predictions = estimator.predict(X)
        return class_pair, predictions

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

    def evaluate_single_classifier(self, X, y, class_pair):
        """
        评估单个二分类器的性能。

        参数:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 标签向量 (n_samples,)
        - class_pair: 要评估的类对 (i, j)

        返回:
        - 该二分类器的准确率
        """
        if class_pair not in self.estimators_:
            raise ValueError(f"Class pair {class_pair} does not exist in the trained classifiers.")

        # 获取属于这两个类别的样本
        mask = np.isin(y, [self.classes_[class_pair[0]], self.classes_[class_pair[1]]])
        X_pair = X[mask]
        y_pair = y[mask]

        # 将类别标签转换为 0 和 1
        y_binary = np.where(y_pair == self.classes_[class_pair[0]], -1, 1)

        # 获取对应的二分类器
        estimator = self.estimators_[class_pair]

        # 进行预测
        y_pred = estimator.predict(X_pair)

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
        for class_pair in self.estimators_:
            accuracy = self.evaluate_single_classifier(X, y, class_pair)
            accuracies[class_pair] = accuracy
            print(f"Accuracy of classifier for class pair {class_pair}: {accuracy:.4f}")

        return accuracies