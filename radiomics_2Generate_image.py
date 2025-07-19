import os
from pathlib import Path
os.environ["SCIPY_ARRAY_API"] = "1"
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
from my_package.draw import *
from my_package.metrics import *


MODEL_CONFIGS = [
    {  # 第一个模型（决策树）
        'model_name': 'DecisionTree',
        'model_class': DecisionTreeClassifier,  # 模型类
        'data_path': "model/影像2.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 9,
        'smote_seed': 2,
        'model_params': {  # 决策树参数
            'criterion': "gini",
            "splitter": "best",
            'max_depth': 18,
            'min_samples_split': 20,
            'min_samples_leaf': 47,
            "max_features": "sqrt",
            "min_weight_fraction_leaf": 0.25961358438289,
            'random_state': 148,
            "max_leaf_nodes": 40,
            "min_impurity_decrease": 0.0214983750627734,
            "class_weight": "balanced",
            "ccp_alpha": 0.0189045334224271,
        }
    },
    {  # 第二个模型（随机森林）
        'model_name': 'RandomForest',
        'model_class': RandomForestClassifier,  # 模型类
        'data_path': "model/影像2.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 6,
        'smote_seed': 84,
        'model_params': {
            "n_estimators":240,
            "max_depth": 94,
            "min_samples_split": 21,
            "min_samples_leaf": 3,
            "max_features": "log2",
            "criterion": "entropy",
            'random_state': 137,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "warm_start": False
        }
    },
    {
        "model_name": "MultilayerPerceptron",
        'model_class': MLPClassifier,
        'data_path': "model/影像2.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 11,
        'smote_seed': 57,
        'model_params': {
            "hidden_layer_sizes": (100, 50),
            "activation": "logistic",
            "solver": "sgd",
            "alpha": 0.00026585997980111,
            "batch_size": 64,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.0602277590448482,
            "max_iter": 900,
            "early_stopping": False,
            "tol": 0.00400451226117344,
            "momentum": 0.638030204531993,
            "n_iter_no_change":15,
            "random_state":176,
        }
    },
    {
        "model_name": "SupportVectorMachine",
        "model_class": SVC,
        'data_path': "model/影像2.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 39,
        'smote_seed': 47,
        'model_params': {
            "C": 0.0460038148452287,
            "kernel": "poly",
            "gamma": "auto",
            "degree": 2,
            "coef0": 8.61008028690965,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "random_state": 66,
            "probability": True,  # 启用预测概率
            "tol": 0.000280212613595925,  # 容忍度
            "max_iter": 50000,  # 最大迭代次数
            "shrinking": True  # 启用收缩算法
        }
    },
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        'data_path': "model/影像2.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 39,
        'smote_seed': 19,
        'model_params': {
            "C": 0.70317398830512,
            "penalty": "l2",
            "solver": "saga",
            "max_iter": 20000,
            "tol": 0.000451186496166719,
            "class_weight":None,
            "fit_intercept": False,
            "random_state": 72,
            "warm_start": False
        }
    },
    {"model_name": "GaussianNB",
     "model_class": GaussianNB,
     'data_path': "model/影像2.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 33,
     'smote_seed': 47,
     'model_params': {
         "var_smoothing": 1.54683201234064E-08, }
     },
]
if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "IMAT-ML"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / "2"  # 等价于 Path("小图/0")
    output_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录

    metrics_list = []
    for config in MODEL_CONFIGS:
        # 加载数据（使用Path确保路径安全）
        df = pd.read_excel(Path(config['data_path']))
        X = df.drop(columns=["ER"])
        y = df["ER"]

        # 训练模型
        metrics, model, (X_test, y_test, y_test_proba, y_test_pred) = train_model(
            config, X, y, config['model_name']
        )
        metrics_list.append(metrics)

        # 保存预测结果（使用f-string和Path拼接）
        combined = pd.concat([metrics.cv_predictions, metrics.test_predictions])
        combined.to_excel(
            output_dir / f"{config['model_name']}_predictions.xlsx",
            index=False
        )

    # 统一使用模型组名称变量（动态生成文件名）
    plot_roc_curve(
        metrics_list,
        str(output_dir / f"{model_group_name}_ROC.pdf"),
        f"{model_group_name} Models ROC Curves"  # 标题也同步更新
    )

    plot_dca_curve(
        metrics_list,
        str(output_dir / f"{model_group_name}_DCA.pdf"),
        f"{model_group_name} Models DCA Curves"
    )

    plot_test_roc_curves(
        metrics_list,
        str(output_dir / f"{model_group_name}_Test_ROC.pdf"),
        f"{model_group_name} Models ROC Curves (Test Set)"
    )
