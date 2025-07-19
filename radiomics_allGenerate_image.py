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
        "model_class": DecisionTreeClassifier,
        "data_path": "model/影像.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 62,
        'smote_seed': 0,
        'model_params': {  # 决策树参数
            "criterion": "entropy",
            "splitter": "best",
            "max_depth": 41,
            "min_samples_split": 32,
            "min_samples_leaf": 20,
            "min_weight_fraction_leaf": 0.100783914970753,
            "max_features": "sqrt",
            "random_state": 187,
            "max_leaf_nodes": 100,
            "min_impurity_decrease": 0.00908438633487539,
            "class_weight": "balanced",
            "ccp_alpha": 0.00879908554985088
        }
    },
    {  # 第二个模型（随机森林）
        'model_name': 'RandomForest',
        "model_class": RandomForestClassifier,
        "data_path": "model/影像.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 73,
        'smote_seed': 43,
        'model_params': {
            "n_estimators": 700,
            "max_depth": 56,
            "min_samples_split": 31,
            "min_samples_leaf": 32,
            "max_features": "log2",
            "criterion": "entropy",
            'random_state': 179,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "warm_start": False
        }
    },
    {
        "model_name": "MultilayerPerceptron",
        "model_class": MLPClassifier,
        "data_path": "model/影像.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 73,
        'smote_seed': 22,
        'model_params': {
            "hidden_layer_sizes": (150,),
            "activation": "logistic",
            "solver": "adam",
            "alpha": 0.00231484989555992,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.000490233186159343,
            "max_iter": 100,
            "early_stopping": False,
            "tol": 0.000442318093857444,
            "momentum": 0.679049269915152,
            "n_iter_no_change": 21,
            "random_state": 101,
        }
    },
    {
        "model_name": "SupportVectorMachine",
        "model_class": SVC,
        "data_path": "model/影像.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 52,
        'smote_seed': 961,
        'model_params': {
            "C": 0.724,
            "kernel": "poly",
            "gamma": "auto",
            "degree": 5,
            "coef0": 0.0025448,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "probability": True,  # 启用预测概率
            "tol": 0.02256,  # 容忍度
            "max_iter": 38000,  # 最大迭代次数
            "shrinking": True,  # 启用收缩算法
            "random_state": 697,
        }
    },
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        "data_path": "model/影像.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 73,
        'smote_seed': 48,
        'model_params': {
            "C": 0.0210345395908164,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 20000,
            "tol": 0.0000393613091322363,
            "class_weight": "balanced",
            "fit_intercept": True,
            "random_state": 55,
            "warm_start": False}
    },
    {"model_name": "GaussianNB",
     "model_class": GaussianNB,
     "data_path": "model/影像.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 40,
     'smote_seed': 57,
     'model_params': {  # 决策树参数
         "var_smoothing": 1.83778336237403E-07
     }
     },
]

if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "SIVS-ML"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / "5"  # 等价于 Path("小图/0")
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
