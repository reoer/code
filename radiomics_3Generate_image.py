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
        'data_path': "model/影像3.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 33,
        'smote_seed': 86,
        'model_params': {  # 决策树参数
            'criterion': "entropy",
            "splitter": "random",
            'max_depth': 34,
            'min_samples_split': 36,
            'min_samples_leaf': 1,
            #"max_features": "sqrt",
            "min_weight_fraction_leaf": 0.159986016384555,
            'random_state': 151,
            "max_leaf_nodes": 75,
            "min_impurity_decrease": 0.000360911331667986,
            "class_weight": None,
            "ccp_alpha": 0.0130585241030417,
        }
    },
    {  # 第二个模型（随机森林）
        'model_name': 'RandomForest',
        'model_class': RandomForestClassifier,  # 模型类
        'data_path': "model/影像3.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 59,
        'smote_seed': 20,
        'model_params': {
            "n_estimators": 100,
            "max_depth": 13,
            "min_samples_split": 23,
            "min_samples_leaf": 17,
            "max_features": "sqrt",
            "criterion": "gini",
            'random_state': 164,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "warm_start": False
        }
    },
    {
        "model_name": "MultilayerPerceptron",
        'model_class': MLPClassifier,
        'data_path': "model/影像3.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 70,
        'smote_seed': 82,
        'model_params': {
            "hidden_layer_sizes": (100,),
            "activation": "logistic",
            "solver": "adam",
            "alpha": 0.000473345151436814,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.0000210014610718522,
            "max_iter": 200,
            "early_stopping": False,
            "tol":0.0000636929200413189,
            "momentum": 0.858819169957582,
            "n_iter_no_change": 20,
            "random_state": 108,
        }
    },
    {
        "model_name": "SupportVectorMachine",
        "model_class": SVC,
        'data_path': "model/影像3.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 42,
        'smote_seed': 52,
        'model_params': {
            "C": 0.0700892737349158,
            "kernel": "poly",
            "gamma": "auto",
            "degree":1,
            "coef0": 4.59024567249963,
            "class_weight": "balanced",  # 使用手动计算的类权重
            "random_state": 5,
            "probability": True,  # 启用预测概率
            "tol": 0.000338988319800448,  # 容忍度
            "max_iter": 47000,  # 最大迭代次数
            "shrinking": False  # 启用收缩算法
        }
    },
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        'data_path': "model/影像3.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed': 59,
        'smote_seed': 45,
        'model_params': {
            "C": 0.0128312314571848,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 20000,
            "tol": 0.0734040835621783,
            "class_weight": "balanced",
            "fit_intercept": True,
            "random_state": 13,
            "warm_start": False
        }
    },
    {"model_name": "GaussianNB",
     "model_class": GaussianNB,
     'data_path': "model/影像3.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 31,
     'smote_seed': 11,
     'model_params': {
         "var_smoothing": 0.000143944470174987, }
     },
]
if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "VAT-ML"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / "3"  # 等价于 Path("小图/0")
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
