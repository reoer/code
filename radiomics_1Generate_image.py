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
        'data_path': "model/影像1.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 60,
        'smote_seed': 95,
        'model_params': {  # 决策树参数
            'criterion': "entropy",
            "splitter": "best",
            'max_depth': 29,
            'min_samples_split': 46,
            'min_samples_leaf': 20,
            "max_features": "log2",
            "min_weight_fraction_leaf": 0.401126589647756,
            'random_state': 189,
            "max_leaf_nodes": 92,
            "min_impurity_decrease": 0.0223303818542394,
            "class_weight": "balanced",
            "ccp_alpha": 0.00769301927268868,
        }
    },
    {  # 第二个模型（随机森林）
        'model_name': 'RandomForest',
        'model_class': RandomForestClassifier,  # 模型类
        'data_path': "model/影像1.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 92,
        'smote_seed': 59,
        'model_params': {
            "n_estimators": 390,
            "max_depth": 45,
            "min_samples_split": 13,
            "min_samples_leaf": 28,
            "max_features": "sqrt",
            "criterion": "entropy",
            'random_state': 100,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "warm_start": False
        }
    },
    {
        "model_name": "MultilayerPerceptron",
        'model_class': MLPClassifier,
        'data_path': "model/影像1.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 86,
        'smote_seed': 70,
        'model_params': {
            "hidden_layer_sizes": (100, 100),
            "activation": "logistic",
            "solver": "lbfgs",
            "alpha": 0.0041753759332789,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.0000835791115569504,
            "max_iter": 200,
            "early_stopping": False,
            "tol": 0.00704836256990867,
            "momentum": 0.557927026986558,
            "n_iter_no_change": 44,
            "random_state": 180,
        }
    },
    {
        "model_name": "SupportVectorMachine",
        "model_class": SVC,
        'data_path': "model/影像1.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 48,
        'smote_seed': 51,
        'model_params': {
            "C": 0.135117558789895,
            "kernel": "poly",
            "gamma": "auto",
            "degree": 1,
            "coef0": 0.47867009327404,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "random_state": 45,
            "probability": True,  # 启用预测概率
            "tol": 0.000636423691103509,  # 容忍度
            "max_iter": 35000,  # 最大迭代次数
            "shrinking": True  # 启用收缩算法
        }
    },
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        'data_path': "model/影像1.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 48,
        'smote_seed': 54,
        'model_params': {
            "C": 6.40348124326391,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 20000,
            "tol": 0.0528812631310282,
            "class_weight": "balanced",
            "fit_intercept": True,
            "random_state": 44,
            "warm_start": False
        }
    },
    {"model_name": "GaussianNB",
     "model_class": GaussianNB,
     'data_path': "model/影像1.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 45,
     'smote_seed': 44,
     'model_params': {
         "var_smoothing": 0.0980242724911948, }
     },
]


if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "SAT-ML"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / "1"  # 等价于 Path("小图/0")
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

