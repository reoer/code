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
        'data_path': "model/影像4.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 73,
        'smote_seed': 7,
        'model_params': {  # 决策树参数
            'criterion': "gini",
            "splitter": "best",
            'max_depth': 19,
            'min_samples_split': 69,
            'min_samples_leaf': 8,
            "max_features": "sqrt",
            "min_weight_fraction_leaf": 0.457827745021064,
            'random_state': 102,
            "max_leaf_nodes": 58,
            "min_impurity_decrease": 0.000227605816076433,
            "class_weight": "balanced",
            "ccp_alpha": 0.00238659855379965,
        }
    },
    {  # 第二个模型（随机森林）
        'model_name': 'RandomForest',
        'model_class': RandomForestClassifier,  # 模型类
        'data_path': "model/影像4.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 25,
        'smote_seed': 88,
        'model_params': {
            "n_estimators": 1000,
            "max_depth": 84,
            "min_samples_split": 31,
            "min_samples_leaf": 30,
            "max_features": "log2",
            "criterion": "gini",
            'random_state': 181,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "warm_start": False
        }
    },
    {
        "model_name": "MultilayerPerceptron",
        'model_class': MLPClassifier,
        'data_path': "model/影像4.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 68,
        'smote_seed': 25,
        'model_params': {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.000209834707046727,
            "batch_size": 64,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.0095613040075855,
            "max_iter": 100,
            "early_stopping": True,
            "tol": 0.0000715151284560171,
            "momentum": 0.851904305811668,
            "n_iter_no_change": 39,
            "random_state": 100,
        }
    },
    {
        "model_name": "SupportVectorMachine",
        "model_class": SVC,
        'data_path': "model/影像4.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 34,
        'smote_seed': 78,
        'model_params': {
            "C": 1.31832702664941,
            "kernel": "poly",
            "gamma": "auto",
            "degree": 8,
            "coef0": 9.12815251704487,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "random_state": 90,
            "probability": True,  # 启用预测概率
            "tol": 0.000746468706446178,  # 容忍度
            "max_iter": 46000,  # 最大迭代次数
            "shrinking": False  # 启用收缩算法
        }
    },
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        'data_path': "model/影像4.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 53,
        'smote_seed': 87,
        'model_params': {
            "C": 0.0100454951986339,
            "penalty": "l2",
            "solver": "newton-cg",
            "max_iter": 20000,
            "tol": 0.0437433482478941,
            "class_weight": "balanced",
            "fit_intercept": True,
            "random_state": 0,
            "warm_start": False
        }
    },
    {"model_name": "GaussianNB",
     "model_class": GaussianNB,
     'data_path': "model/影像4.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 34,
     'smote_seed': 64,
     'model_params': {
         "var_smoothing": 4.27578035674892E-07, }
     },
]
if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "SM-ML"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / "4"  # 等价于 Path("小图/0")
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