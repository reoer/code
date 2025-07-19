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
        'data_path': "model/影像0.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 43,
        'smote_seed': 49,
        'model_params': {  # 决策树参数
            'criterion': "entropy",
            "splitter": "best",
            'max_depth': 18,
            'min_samples_split': 16,
            'min_samples_leaf': 5,
            "min_weight_fraction_leaf": 0.00599632681306425,
            "max_features": "sqrt",
            'random_state': 122,
            "max_leaf_nodes": 44,
            "min_impurity_decrease": 0.0177578252094426,
            "class_weight": "balanced",
            "ccp_alpha": 0.0263820819266371,
        }
    },
    {  # 第二个模型（随机森林）
        'model_name': 'RandomForest',
        'model_class': RandomForestClassifier,  # 模型类
        'data_path': "model/影像0.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 24,
        'smote_seed': 30,
        'model_params': {
            "n_estimators": 650,
            "max_depth": 16,
            "min_samples_split": 19,
            "min_samples_leaf": 22,
            "max_features": "sqrt",
            "criterion": "entropy",
            'random_state': 131,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "warm_start": False
        }
    },
    {
        "model_name": "MultilayerPerceptron",
        'model_class': MLPClassifier,
        'data_path': "model/影像0.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 45,
        'smote_seed': 54,
        'model_params': {
            "hidden_layer_sizes": (100, 50),
            "activation": "logistic",
            "solver": "adam",
            "alpha": 0.00038,
            "batch_size": "auto",
            "learning_rate": "invscaling",
            "learning_rate_init": 0.00156279332342121,
            "max_iter": 700,
            "early_stopping": False,
            "tol": 0.07966,
            "momentum": 0.891513898950495,
            "n_iter_no_change": 26,
            "random_state": 136,
        }
    },
    {
        "model_name": "SupportVectorMachine",
        "model_class": SVC,
        'data_path': "model/影像0.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 61,
        'smote_seed': 51,
        'model_params': {
            "C": 0.162550427767025,
            "kernel": "poly",
            "gamma": "auto",
            "degree": 3,
            "coef0": 0.00570277269417124,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "random_state": 56,
            "probability": True,  # 启用预测概率
            "tol": 0.00016372695731776,  # 容忍度
            "max_iter": 50000,  # 最大迭代次数
            "shrinking": False  # 启用收缩算法
        }
    },
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        'data_path': "model/影像0.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 55,
        'smote_seed': 97,
        'model_params': {
            "C": 0.0376143192131473,
            "penalty": "l2",
            "solver": "newton-cg",
            "max_iter": 20000,
            "tol": 0.0628049127852337,
            "class_weight": "balanced",
            "fit_intercept": True,
            "random_state": 92,
            "warm_start": False
        }
    },
    {"model_name": "GaussianNB",
     "model_class": GaussianNB,
     'data_path': "model/影像0.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 58,
     'smote_seed': 41,
     'model_params': {
         "var_smoothing": 0.000842427573204091, }
     },
]

if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "ALL-ML"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / "0"  # 等价于 Path("小图/0")
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