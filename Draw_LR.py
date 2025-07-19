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
    {"model_name": "Clinic",
     "model_class": LogisticRegression,
     "data_path": "model/临床.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 77,
     'smote_seed': 67,
     'model_params': {
         "C": 0.687965522,
         "penalty": "l2",
         "solver": "lbfgs",
         "max_iter": 20000,
         "tol": 0.0267169822007772,
         "class_weight": "balanced",
         "fit_intercept": True,
         "random_state": 40,
         "warm_start": False}
     },
    {"model_name": "CT",
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
    {"model_name": "CT+Clinic",
     "model_class": LogisticRegression,
     "data_path": "model/融合.xlsx",
     'test_size': 0.3,
     'n_splits': 5,
     'split_seed': 1027,
     'cv_seed': 93,
     'smote_seed': 60,
     'model_params': {
         "C": 0.0638981610350925,
         "penalty": "l2",
         "solver": "liblinear",
         "max_iter": 20000,
         "tol": 0.000146767899745637,
         "class_weight": "balanced",
         "fit_intercept": True,
         "random_state": 16,
         "warm_start": False}
     },
]
if __name__ == "__main__":
    # 定义常量（可轻松修改为其他模型组名称）
    model_group_name = "LogisticRegression"  # 只需修改此处即可全局生效
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
