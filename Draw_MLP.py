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

    {
        "model_name": "Clinic",
        "model_class": MLPClassifier,
        "data_path": "model/临床.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed':1027,
        'cv_seed': 18,
        'smote_seed': 42,
        'model_params': {
            "hidden_layer_sizes": (50,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.00857329106121873,
            "batch_size": 16,
            "learning_rate": "constant",
            "learning_rate_init": 0.0316692674007109,
            "max_iter": 1000,
            "early_stopping": False,
            "tol": 0.00635426903184179,
            "momentum": 0.723225948603886,
            "n_iter_no_change": 40,
            "random_state": 139,
        }
    },
    {
        "model_name": "CT",
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
            "tol":0.000442318093857444,
            "momentum": 0.679049269915152,
            "n_iter_no_change": 21,
            "random_state": 101,
        }
    },
    {
        "model_name": "CT+Clinic",
        "model_class": MLPClassifier,
        "data_path": "model/融合.xlsx",
        'test_size': 0.3,
        'n_splits': 5,
        'split_seed': 1027,
        'cv_seed':40,
        'smote_seed': 87,
        'model_params': {
            "hidden_layer_sizes": (100, 100),
            "activation": "logistic",
            "solver": "adam",
            "alpha": 0.00733656116592867,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.034152834162218,
            "max_iter": 600,
            "early_stopping": True,
            "tol": 0.0106055109521209,
            "momentum": 0.759564898136696,
            "n_iter_no_change": 27,
            "random_state": 182,
        }
    },
]


if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "MLP"  # 只需修改此处即可全局生效
    # 定义输出目录（使用Path对象确保跨平台兼容）
    output_dir = Path("model") / model_group_name  # 等价于 Path("小图/0")
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
