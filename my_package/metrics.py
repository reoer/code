import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class MetricsContainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.conf_matrix_list = []
        self.smote_auc_scores = []
        self.smote_f1_scores = []
        self.fpr_list = []
        self.tpr_list = []
        self.roc_auc_list = []
        self.net_benefit_test = []
        self.net_benefit_all_positive = []
        self.test_fpr = None
        self.test_tpr = None
        self.test_auc = 0.0
        self.cv_predictions = []
        self.test_predictions = None

    def calculate_test_roc(self, y_test, y_test_proba):
        self.test_fpr, self.test_tpr, _ = roc_curve(y_test, y_test_proba)
        self.test_auc = auc(self.test_fpr, self.test_tpr)

def calculate_net_benefit(y_true, y_pred_proba, thresholds):
    net_benefit = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        n = len(y_true)
        net_benefit.append((tp / n) - (fp / n) * (threshold / (1 - threshold)))
    return np.array(net_benefit)

def train_model(config, X, y, model_name):
    ids = X["subfolder"]  # 假设数据中包含ID列
    metrics = MetricsContainer(model_name)
    X = X.drop(columns=["subfolder"])
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids,
        test_size=config['test_size'],
        random_state=config['split_seed'],
        stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 应用 SMOTE
    smtk = SMOTETomek(random_state=config['smote_seed'])
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 创建参数副本防止修改原始配置
    model_params = config['model_params'].copy()  # 重要！先复制原始参数

    # 动态加载模型类
    model_class = config['model_class']
    model = model_class(**model_params)  # 使用更新后的参数
    # 数据标准化
    scaler = StandardScaler()
    skf = StratifiedKFold(
        n_splits=config['n_splits'],
        random_state=config['cv_seed'],
        shuffle=True
    )
    fold_number = 0
    for train_index, val_index in skf.split(X_train, y_train):  # 使用原始训练集划分
        fold_number += 1
        # 获取原始训练集和验证集的ID
        id_val_fold = id_train.iloc[val_index].values

        # 对训练部分进行SMOTE
        X_train_fold_raw = X_train.iloc[train_index]
        y_train_fold_raw = y_train.iloc[train_index]
        X_train_fold_smtk, y_train_fold_smtk = smtk.fit_resample(X_train_fold_raw, y_train_fold_raw)

        # 验证集保持原始数据
        X_val_fold = X_train.iloc[val_index]
        y_val_fold = y_train.iloc[val_index]

        # 标准化数据
        X_train_fold_smtk_scaled = scaler.fit_transform(X_train_fold_smtk)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # 训练模型
        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)  # 修正：传入特征矩阵 X_val_fold_scaled

        # 计算指标
        metrics.smote_auc_scores.append(roc_auc_score(y_val_fold, y_val_proba))
        metrics.smote_f1_scores.append(f1_score(y_val_fold, y_val_pred))
        metrics.conf_matrix_list.append(confusion_matrix(y_val_fold, y_val_pred))

        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        metrics.fpr_list.append(fpr)
        metrics.tpr_list.append(tpr)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        metrics.roc_auc_list.append(auc(fpr, tpr))
        y_pred = (y_val_proba >= optimal_threshold).astype(int)
        # 保存结果时添加ID
        fold_df = pd.DataFrame({
            "Predictedvalue": y_val_proba,
            'Predicted': y_pred,
            'Actual': y_val_fold.values,
            'Fold': fold_number,
            'subfolder': id_val_fold  # 新增ID列
        })
        metrics.cv_predictions.append(fold_df)

    # 合并所有交叉验证结果
    metrics.cv_predictions = pd.concat(metrics.cv_predictions)

    # 测试集评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)

    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)
    metrics.calculate_test_roc(y_test, y_test_proba)

    # 计算 DCA 指标
    thresholds = np.linspace(0, 1, 100)
    metrics.net_benefit_test = calculate_net_benefit(y_test, y_test_proba, thresholds)
    metrics.net_benefit_all_positive = np.full_like(thresholds, (y_test.sum() / len(y_test)) - (
            (1 - y_test).sum() / len(y_test)) * (thresholds / (1 - thresholds)))
    fpr2, tpr2, thresholds = roc_curve(y_test, y_test_proba)
    youden_index2 = tpr2 - fpr2
    optimal_idx2 = np.argmax(youden_index2)
    optimal_threshold2 = thresholds[optimal_idx2]
    metrics.roc_auc_list.append(auc(fpr2, tpr2))
    y_pred2 = (y_test_proba >= optimal_threshold2).astype(int)
    # 测试集结果
    test_df = pd.DataFrame({
        "Predictedvalue": y_test_proba,
        'Predicted': y_pred2,
        'Actual': y_test.values,
        'Fold': 0,
        'subfolder': id_test.values  # 新增测试集ID
    })
    metrics.test_predictions = test_df

    return metrics, model, (X_test, y_test, y_test_proba, y_test_pred)