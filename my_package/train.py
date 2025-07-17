import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

os.environ["SCIPY_ARRAY_API"] = "1"
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, brier_score_loss
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK

random_state_ini = 1027


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # 裁剪预测值以防止 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # 计算交叉熵损失
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def objective_SVC(params, X, y, test_size=0.3, split_k=5, auc_results=None):
    """
    支持超参数优化的目标函数

    参数:
    params: hyperopt生成的参数字典
    X: 特征数据 (DataFrame)
    y: 目标变量 (Series)
    test_size: 测试集比例 (默认0.3)
    split_k: 交叉验证折数 (默认5)
    auc_results: 用于存储结果的列表 (可选)

    返回:
    符合hyperopt要求的字典 (包含'loss'和'status')
    """
    random_state_split = random_state_ini
    random_state_cv = int(params['random_state_cv'])
    random_state_smote = int(params['random_state_smote'])
    random_state_model = int(params['random_state_model'])

    # 重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split, stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 过采样处理
    smtk = SMOTETomek(random_state=random_state_smote)
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 初始化模型
    model = SVC(
        C=params['C'],
        kernel=params['kernel'],
        gamma=params['gamma'],
        degree=int(params['degree']),
        coef0=params['coef0'],
        class_weight="balanced",
        random_state=random_state_model,
        probability=True,
        tol=params['tol'],
        max_iter=int(params['max_iter']),
        shrinking=params['shrinking']
    )

    # 初始化指标收集
    smote_auc_scores = []
    smote_f1_scores = []
    smote_sensitivity_scores = []
    smote_specificity_scores = []
    smote_brier_scores = []
    best_thresholds = []
    all_val_probas = []
    all_val_y = []

    # 标准化器和交叉验证
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=split_k, random_state=random_state_cv, shuffle=True)

    # 交叉验证
    LOSS = []
    for train_index, val_index in skf.split(X_train, y_train):
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

        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)

        # 计算各种指标
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)

        y_val_pred_best = (y_val_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fold_f1 = f1_score(y_val_fold, y_val_pred_best)
        fold_brier = brier_score_loss(y_val_fold, y_val_proba)

        smote_sensitivity_scores.append(sensitivity)
        smote_specificity_scores.append(specificity)
        smote_f1_scores.append(fold_f1)
        smote_brier_scores.append(fold_brier)
        all_val_probas.extend(y_val_proba)
        all_val_y.extend(y_val_fold)

        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        smote_auc_scores.append(fold_auc)
        loss = binary_cross_entropy(y_val_fold, y_val_proba)
        LOSS.append(loss)

    # 计算全局最佳阈值
    fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
    youden_index_all = tpr_all - fpr_all
    global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

    # 计算各种平均值
    auc_scores_min = np.min(smote_auc_scores)
    mean_auc = np.mean(smote_auc_scores)
    mean_f1 = np.mean(smote_f1_scores)
    mean_sensitivity = np.mean(smote_sensitivity_scores)
    mean_specificity = np.mean(smote_specificity_scores)
    mean_brier = np.mean(smote_brier_scores)

    # 对测试集进行标准化和评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)

    # 测试集指标计算
    y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
    test_sensitivity = tp_test / (tp_test + fn_test)
    test_specificity = tn_test / (tn_test + fp_test)
    test_f1 = f1_score(y_test, y_test_pred_best)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    all_loss = sum(LOSS)

    # 构建结果字典
    result_dict = {
        'C': params['C'],
        'kernel': params['kernel'],
        'gamma': params['gamma'],
        'degree': int(params['degree']),
        'coef0': params['coef0'],
        'class_weight': "balanced",
        'tol': params['tol'],
        'max_iter': int(params['max_iter']),
        'shrinking': params['shrinking'],
        'random_state_split': random_state_split,
        'random_state_cv': random_state_cv,
        'random_state_smote': random_state_smote,
        'random_state_model': random_state_model,
        "smote_auc_scores": smote_auc_scores,
        "auc_scores_min": auc_scores_min,
        'mean_auc': mean_auc,
        'test_auc': test_auc,
        'mean_f1': mean_f1,
        'test_f1': test_f1,
        'best_threshold': global_best_threshold,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'mean_brier': mean_brier,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_brier': test_brier,
        'smote_sensitivity_scores': smote_sensitivity_scores,
        'smote_specificity_scores': smote_specificity_scores,
        'smote_brier_scores': smote_brier_scores,
        'auc_product': mean_auc * test_auc,
        'f1_product': mean_f1 * test_f1,
        'youhua': mean_auc * test_auc * mean_f1 * test_f1,
        "loss": all_loss,
        "SORT": (mean_auc > 0.7) & (test_auc > 0.7),
        "brier": mean_brier + test_brier,
    }

    # 如果提供了auc_results列表，则将结果存入
    if auc_results is not None:
        auc_results.append(result_dict)

    # 返回hyperopt需要的格式
    return {'loss': all_loss, 'status': STATUS_OK, 'result_dict': result_dict}


def objective_MLP(params, X, y, test_size=0.3, split_k=5, auc_results=None):
    """
    支持超参数优化的目标函数

    参数:
    params: hyperopt生成的参数字典
    X: 特征数据 (DataFrame)
    y: 目标变量 (Series)
    test_size: 测试集比例 (默认0.3)
    split_k: 交叉验证折数 (默认5)
    auc_results: 用于存储结果的列表 (可选)

    返回:
    符合hyperopt要求的字典 (包含'loss'和'status')
    """
    random_state_split = random_state_ini
    random_state_cv = int(params['random_state_cv'])
    random_state_smote = int(params['random_state_smote'])
    random_state_model = int(params['random_state_model'])

    # 重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split, stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 过采样处理
    smtk = SMOTETomek(random_state=random_state_smote)
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 初始化模型
    model = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=int(params['max_iter']),
        early_stopping=params['early_stopping'],
        tol=params['tol'],
        momentum=params['momentum'],
        n_iter_no_change=int(params['n_iter_no_change']),
        random_state=random_state_model,
    )

    # 初始化指标收集
    smote_auc_scores = []
    smote_f1_scores = []
    smote_sensitivity_scores = []
    smote_specificity_scores = []
    smote_brier_scores = []
    best_thresholds = []
    all_val_probas = []
    all_val_y = []

    # 标准化器和交叉验证
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=split_k, random_state=random_state_cv, shuffle=True)

    # 交叉验证
    LOSS = []
    for train_index, val_index in skf.split(X_train, y_train):
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

        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)

        # 计算各种指标
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)

        y_val_pred_best = (y_val_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fold_f1 = f1_score(y_val_fold, y_val_pred_best)
        fold_brier = brier_score_loss(y_val_fold, y_val_proba)

        smote_sensitivity_scores.append(sensitivity)
        smote_specificity_scores.append(specificity)
        smote_f1_scores.append(fold_f1)
        smote_brier_scores.append(fold_brier)
        all_val_probas.extend(y_val_proba)
        all_val_y.extend(y_val_fold)

        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        smote_auc_scores.append(fold_auc)
        loss = binary_cross_entropy(y_val_fold, y_val_proba)
        LOSS.append(loss)

    # 计算全局最佳阈值
    fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
    youden_index_all = tpr_all - fpr_all
    global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

    # 计算各种平均值
    auc_scores_min = np.min(smote_auc_scores)
    mean_auc = np.mean(smote_auc_scores)
    mean_f1 = np.mean(smote_f1_scores)
    mean_sensitivity = np.mean(smote_sensitivity_scores)
    mean_specificity = np.mean(smote_specificity_scores)
    mean_brier = np.mean(smote_brier_scores)

    # 对测试集进行标准化和评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)

    # 测试集指标计算
    y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
    test_sensitivity = tp_test / (tp_test + fn_test)
    test_specificity = tn_test / (tn_test + fp_test)
    test_f1 = f1_score(y_test, y_test_pred_best)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    all_loss = sum(LOSS)

    # 构建结果字典
    result_dict = {
        'hidden_layer_sizes': params['hidden_layer_sizes'],
        'activation': params['activation'],
        'solver': params['solver'],
        'alpha': params['alpha'],
        'batch_size': params['batch_size'],
        'learning_rate': params['learning_rate'],
        'learning_rate_init': params['learning_rate_init'],
        'max_iter': int(params['max_iter']),
        'early_stopping': params['early_stopping'],
        'tol': params['tol'],
        'momentum': params['momentum'],
        'n_iter_no_change': int(params['n_iter_no_change']),
        'random_state_split': random_state_split,
        'random_state_cv': random_state_cv,
        'random_state_smote': random_state_smote,
        'random_state_model': random_state_model,
        "smote_auc_scores": smote_auc_scores,
        "auc_scores_min": auc_scores_min,
        'mean_auc': mean_auc,
        'test_auc': test_auc,
        'mean_f1': mean_f1,
        'test_f1': test_f1,
        # ==========================================
        'best_threshold': global_best_threshold,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'mean_brier': mean_brier,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_brier': test_brier,
        'smote_sensitivity_scores': smote_sensitivity_scores,
        'smote_specificity_scores': smote_specificity_scores,
        'smote_brier_scores': smote_brier_scores,
        # ==========================================
        'auc_product': mean_auc * test_auc,
        'f1_product': mean_f1 * test_f1,
        'youhua': mean_auc * test_auc * mean_f1 * test_f1,
        "loss": all_loss,
        "SORT": (mean_auc > 0.7) & (test_auc > 0.7),
        "brier": mean_brier + test_brier,
    }

    # 如果提供了auc_results列表，则将结果存入
    if auc_results is not None:
        auc_results.append(result_dict)

    # 返回hyperopt需要的格式
    return {'loss': all_loss, 'status': STATUS_OK, 'result_dict': result_dict}


def objective_DT(params, X, y, test_size=0.3, split_k=5, auc_results=None):
    """
    支持超参数优化的目标函数

    参数:
    params: hyperopt生成的参数字典
    X: 特征数据 (DataFrame)
    y: 目标变量 (Series)
    test_size: 测试集比例 (默认0.3)
    split_k: 交叉验证折数 (默认5)
    auc_results: 用于存储结果的列表 (可选)

    返回:
    符合hyperopt要求的字典 (包含'loss'和'status')
    """
    random_state_split = random_state_ini
    random_state_cv = int(params['random_state_cv'])
    random_state_smote = int(params['random_state_smote'])
    random_state_model = int(params['random_state_model'])

    # 重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split, stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 过采样处理
    smtk = SMOTETomek(random_state=random_state_smote)
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 初始化模型
    model = DecisionTreeClassifier(
        criterion=params['criterion'],
        splitter=params['splitter'],
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
        max_features=params['max_features'],
        random_state=int(params['random_state_model']),
        max_leaf_nodes=int(params['max_leaf_nodes']),
        min_impurity_decrease=params['min_impurity_decrease'],
        class_weight=params['class_weight'],
        ccp_alpha=params['ccp_alpha']
    )

    # 初始化指标收集
    smote_auc_scores = []
    smote_f1_scores = []
    smote_sensitivity_scores = []
    smote_specificity_scores = []
    smote_brier_scores = []
    best_thresholds = []
    all_val_probas = []
    all_val_y = []

    # 标准化器和交叉验证
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=split_k, random_state=random_state_cv, shuffle=True)

    # 交叉验证
    LOSS = []
    for train_index, val_index in skf.split(X_train, y_train):
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

        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)

        # 计算各种指标
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)

        y_val_pred_best = (y_val_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fold_f1 = f1_score(y_val_fold, y_val_pred_best)
        fold_brier = brier_score_loss(y_val_fold, y_val_proba)

        smote_sensitivity_scores.append(sensitivity)
        smote_specificity_scores.append(specificity)
        smote_f1_scores.append(fold_f1)
        smote_brier_scores.append(fold_brier)
        all_val_probas.extend(y_val_proba)
        all_val_y.extend(y_val_fold)

        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        smote_auc_scores.append(fold_auc)
        loss = binary_cross_entropy(y_val_fold, y_val_proba)
        LOSS.append(loss)

    # 计算全局最佳阈值
    fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
    youden_index_all = tpr_all - fpr_all
    global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

    # 计算各种平均值
    auc_scores_min = np.min(smote_auc_scores)
    mean_auc = np.mean(smote_auc_scores)
    mean_f1 = np.mean(smote_f1_scores)
    mean_sensitivity = np.mean(smote_sensitivity_scores)
    mean_specificity = np.mean(smote_specificity_scores)
    mean_brier = np.mean(smote_brier_scores)

    # 对测试集进行标准化和评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)

    # 测试集指标计算
    y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
    test_sensitivity = tp_test / (tp_test + fn_test)
    test_specificity = tn_test / (tn_test + fp_test)
    test_f1 = f1_score(y_test, y_test_pred_best)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    all_loss = sum(LOSS)

    # 构建结果字典
    result_dict = {
        'criterion': params["criterion"],
        "splitter": params["splitter"],
        'max_depth': int(params['max_depth']),
        "min_samples_split": int(params['min_samples_split']),
        "min_samples_leaf": int(params['min_samples_leaf']),
        "min_weight_fraction_leaf": params['min_weight_fraction_leaf'],
        "max_features": params['max_features'],
        "random_state": params['random_state_model'],
        "max_leaf_nodes": params['max_leaf_nodes'],
        "min_impurity_decrease": params['min_impurity_decrease'],
        "class_weight": params['class_weight'],
        "ccp_alpha": params["ccp_alpha"],
        'random_state_split': random_state_split,
        'random_state_cv': random_state_cv,
        'random_state_smote': random_state_smote,
        'random_state_model': random_state_model,
        "smote_auc_scores": smote_auc_scores,
        "auc_scores_min": auc_scores_min,
        'mean_auc': mean_auc,
        'test_auc': test_auc,
        'mean_f1': mean_f1,
        'test_f1': test_f1,
        # =============== 新增指标存储 ===============
        'best_threshold': global_best_threshold,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'mean_brier': mean_brier,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_brier': test_brier,
        'smote_sensitivity_scores': smote_sensitivity_scores,
        'smote_specificity_scores': smote_specificity_scores,
        'smote_brier_scores': smote_brier_scores,
        # ==========================================
        'auc_product': mean_auc * test_auc,
        'f1_product': mean_f1 * test_f1,
        'youhua': mean_auc * test_auc * mean_f1 * test_f1,
        "loss": all_loss,
        "SORT": (mean_auc > 0.7) & (test_auc > 0.7),
        "brier": mean_brier + test_brier,
    }

    # 如果提供了auc_results列表，则将结果存入
    if auc_results is not None:
        auc_results.append(result_dict)

    # 返回hyperopt需要的格式
    return {'loss': all_loss, 'status': STATUS_OK, 'result_dict': result_dict}


def objective_RF(params, X, y, test_size=0.3, split_k=5, auc_results=None):
    """
    支持超参数优化的目标函数

    参数:
    params: hyperopt生成的参数字典
    X: 特征数据 (DataFrame)
    y: 目标变量 (Series)
    test_size: 测试集比例 (默认0.3)
    split_k: 交叉验证折数 (默认5)
    auc_results: 用于存储结果的列表 (可选)

    返回:
    符合hyperopt要求的字典 (包含'loss'和'status')
    """
    random_state_split = random_state_ini
    random_state_cv = int(params['random_state_cv'])
    random_state_smote = int(params['random_state_smote'])
    random_state_model = int(params['random_state_model'])

    # 重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split, stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 过采样处理
    smtk = SMOTETomek(random_state=random_state_smote)
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 初始化模型
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        criterion=params['criterion'],
        random_state=random_state_model,
        class_weight="balanced",  # 使用手动计算的类权重
        warm_start=False
    )

    # 初始化指标收集
    smote_auc_scores = []
    smote_f1_scores = []
    smote_sensitivity_scores = []
    smote_specificity_scores = []
    smote_brier_scores = []
    best_thresholds = []
    all_val_probas = []
    all_val_y = []

    # 标准化器和交叉验证
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=split_k, random_state=random_state_cv, shuffle=True)

    # 交叉验证
    LOSS = []
    for train_index, val_index in skf.split(X_train, y_train):
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

        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)

        # 计算各种指标
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)

        y_val_pred_best = (y_val_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fold_f1 = f1_score(y_val_fold, y_val_pred_best)
        fold_brier = brier_score_loss(y_val_fold, y_val_proba)

        smote_sensitivity_scores.append(sensitivity)
        smote_specificity_scores.append(specificity)
        smote_f1_scores.append(fold_f1)
        smote_brier_scores.append(fold_brier)
        all_val_probas.extend(y_val_proba)
        all_val_y.extend(y_val_fold)

        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        smote_auc_scores.append(fold_auc)
        loss = binary_cross_entropy(y_val_fold, y_val_proba)
        LOSS.append(loss)

    # 计算全局最佳阈值
    fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
    youden_index_all = tpr_all - fpr_all
    global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

    # 计算各种平均值
    auc_scores_min = np.min(smote_auc_scores)
    mean_auc = np.mean(smote_auc_scores)
    mean_f1 = np.mean(smote_f1_scores)
    mean_sensitivity = np.mean(smote_sensitivity_scores)
    mean_specificity = np.mean(smote_specificity_scores)
    mean_brier = np.mean(smote_brier_scores)

    # 对测试集进行标准化和评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)

    # 测试集指标计算
    y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
    test_sensitivity = tp_test / (tp_test + fn_test)
    test_specificity = tn_test / (tn_test + fp_test)
    test_f1 = f1_score(y_test, y_test_pred_best)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    all_loss = sum(LOSS)

    # 构建结果字典
    result_dict = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'max_features': params['max_features'],
        'criterion': params['criterion'],
        'class_weight': params['class_weight'],
        'random_state_split': random_state_split,
        'random_state_cv': random_state_cv,
        'random_state_smote': random_state_smote,
        'random_state_model': random_state_model,
        "smote_auc_scores": smote_auc_scores,
        "auc_scores_min": auc_scores_min,
        'mean_auc': mean_auc,
        'test_auc': test_auc,
        'mean_f1': mean_f1,
        'test_f1': test_f1,
        # =============== 新增指标存储 ===============
        'best_threshold': global_best_threshold,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'mean_brier': mean_brier,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_brier': test_brier,
        'smote_sensitivity_scores': smote_sensitivity_scores,
        'smote_specificity_scores': smote_specificity_scores,
        'smote_brier_scores': smote_brier_scores,
        # ==========================================
        'auc_product': mean_auc * test_auc,
        'f1_product': mean_f1 * test_f1,
        'youhua': mean_auc * test_auc * mean_f1 * test_f1,
        "loss": all_loss,
        "SORT": (mean_auc > 0.7) & (test_auc > 0.7),
        "brier": mean_brier + test_brier,
    }

    # 如果提供了auc_results列表，则将结果存入
    if auc_results is not None:
        auc_results.append(result_dict)

    # 返回hyperopt需要的格式
    return {'loss': all_loss, 'status': STATUS_OK, 'result_dict': result_dict}


def objective_LR(params, X, y, test_size=0.3, split_k=5, auc_results=None):
    """
    支持超参数优化的目标函数

    参数:
    params: hyperopt生成的参数字典
    X: 特征数据 (DataFrame)
    y: 目标变量 (Series)
    test_size: 测试集比例 (默认0.3)
    split_k: 交叉验证折数 (默认5)
    auc_results: 用于存储结果的列表 (可选)

    返回:
    符合hyperopt要求的字典 (包含'loss'和'status')
    """
    random_state_split = random_state_ini
    random_state_cv = int(params['random_state_cv'])
    random_state_smote = int(params['random_state_smote'])
    random_state_model = int(params['random_state_model'])

    # 重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split, stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 过采样处理
    smtk = SMOTETomek(random_state=random_state_smote)
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 初始化模型
    model = LogisticRegression(
        C=params['C'],
        penalty=params['penalty'],
        solver=params['solver'],
        max_iter=20000,  # int(params['max_iter']),
        tol=params['tol'],
        class_weight="balanced",
        fit_intercept=params['fit_intercept'],
        random_state=random_state_model,
        warm_start=False
    )

    # 初始化指标收集
    smote_auc_scores = []
    smote_f1_scores = []
    smote_sensitivity_scores = []
    smote_specificity_scores = []
    smote_brier_scores = []
    best_thresholds = []
    all_val_probas = []
    all_val_y = []

    # 标准化器和交叉验证
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=split_k, random_state=random_state_cv, shuffle=True)

    # 交叉验证
    LOSS = []
    for train_index, val_index in skf.split(X_train, y_train):
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

        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)

        # 计算各种指标
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)

        y_val_pred_best = (y_val_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fold_f1 = f1_score(y_val_fold, y_val_pred_best)
        fold_brier = brier_score_loss(y_val_fold, y_val_proba)

        smote_sensitivity_scores.append(sensitivity)
        smote_specificity_scores.append(specificity)
        smote_f1_scores.append(fold_f1)
        smote_brier_scores.append(fold_brier)
        all_val_probas.extend(y_val_proba)
        all_val_y.extend(y_val_fold)

        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        smote_auc_scores.append(fold_auc)
        loss = binary_cross_entropy(y_val_fold, y_val_proba)
        LOSS.append(loss)

    # 计算全局最佳阈值
    fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
    youden_index_all = tpr_all - fpr_all
    global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

    # 计算各种平均值
    auc_scores_min = np.min(smote_auc_scores)
    mean_auc = np.mean(smote_auc_scores)
    mean_f1 = np.mean(smote_f1_scores)
    mean_sensitivity = np.mean(smote_sensitivity_scores)
    mean_specificity = np.mean(smote_specificity_scores)
    mean_brier = np.mean(smote_brier_scores)

    # 对测试集进行标准化和评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)

    # 测试集指标计算
    y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
    test_sensitivity = tp_test / (tp_test + fn_test)
    test_specificity = tn_test / (tn_test + fp_test)
    test_f1 = f1_score(y_test, y_test_pred_best)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    all_loss = sum(LOSS)

    # 构建结果字典
    result_dict = {
        'C': params['C'],
        'penalty': params['penalty'],
        'solver': params['solver'],
        'max_iter': 20000,  #int(params['max_iter']),  # 新增的 max_iter 参数
        'tol': params['tol'],  # 新增的 tol 参数
        'fit_intercept': params['fit_intercept'],  # 新增的 fit_intercept 参数
        'random_state_split': random_state_split,
        'random_state_cv': random_state_cv,
        'random_state_smote': random_state_smote,
        'random_state_model': random_state_model,
        "smote_auc_scores": smote_auc_scores,
        "auc_scores_min": auc_scores_min,
        'mean_auc': mean_auc,
        'test_auc': test_auc,
        'mean_f1': mean_f1,
        'test_f1': test_f1,
        # =============== 新增指标存储 ===============
        'best_threshold': global_best_threshold,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'mean_brier': mean_brier,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_brier': test_brier,
        'smote_sensitivity_scores': smote_sensitivity_scores,
        'smote_specificity_scores': smote_specificity_scores,
        'smote_brier_scores': smote_brier_scores,
        # ==========================================
        'auc_product': mean_auc * test_auc,
        'f1_product': mean_f1 * test_f1,
        'youhua': mean_auc * test_auc * mean_f1 * test_f1,
        "loss": all_loss,
        "SORT": (mean_auc > 0.7) & (test_auc > 0.7),
        "brier": mean_brier + test_brier,
    }

    # 如果提供了auc_results列表，则将结果存入
    if auc_results is not None:
        auc_results.append(result_dict)

    # 返回hyperopt需要的格式
    return {'loss': all_loss, 'status': STATUS_OK, 'result_dict': result_dict}


def objective_NB(params, X, y, test_size=0.3, split_k=5, auc_results=None):
    """
    支持超参数优化的目标函数

    参数:
    params: hyperopt生成的参数字典
    X: 特征数据 (DataFrame)
    y: 目标变量 (Series)
    test_size: 测试集比例 (默认0.3)
    split_k: 交叉验证折数 (默认5)
    auc_results: 用于存储结果的列表 (可选)

    返回:
    符合hyperopt要求的字典 (包含'loss'和'status')
    """
    random_state_split = random_state_ini
    random_state_cv = int(params['random_state_cv'])
    random_state_smote = int(params['random_state_smote'])

    # 重新划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split, stratify=y
    )

    # 数据插补
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # 过采样处理
    smtk = SMOTETomek(random_state=random_state_smote)
    X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)

    # 初始化模型
    model = GaussianNB(var_smoothing=params['var_smoothing'])

    # 初始化指标收集
    smote_auc_scores = []
    smote_f1_scores = []
    smote_sensitivity_scores = []
    smote_specificity_scores = []
    smote_brier_scores = []
    best_thresholds = []
    all_val_probas = []
    all_val_y = []

    # 标准化器和交叉验证
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=split_k, random_state=random_state_cv, shuffle=True)

    # 交叉验证
    LOSS = []
    for train_index, val_index in skf.split(X_train, y_train):
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

        model.fit(X_train_fold_smtk_scaled, y_train_fold_smtk)
        y_val_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
        y_val_pred = model.predict(X_val_fold_scaled)

        # 计算各种指标
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)

        y_val_pred_best = (y_val_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fold_f1 = f1_score(y_val_fold, y_val_pred_best)
        fold_brier = brier_score_loss(y_val_fold, y_val_proba)

        smote_sensitivity_scores.append(sensitivity)
        smote_specificity_scores.append(specificity)
        smote_f1_scores.append(fold_f1)
        smote_brier_scores.append(fold_brier)
        all_val_probas.extend(y_val_proba)
        all_val_y.extend(y_val_fold)

        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        smote_auc_scores.append(fold_auc)
        loss = binary_cross_entropy(y_val_fold, y_val_proba)
        LOSS.append(loss)

    # 计算全局最佳阈值
    fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
    youden_index_all = tpr_all - fpr_all
    global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

    # 计算各种平均值
    auc_scores_min = np.min(smote_auc_scores)
    mean_auc = np.mean(smote_auc_scores)
    mean_f1 = np.mean(smote_f1_scores)
    mean_sensitivity = np.mean(smote_sensitivity_scores)
    mean_specificity = np.mean(smote_specificity_scores)
    mean_brier = np.mean(smote_brier_scores)

    # 对测试集进行标准化和评估
    X_test_scaled = scaler.transform(X_test)
    X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
    model.fit(X_train_smtk_scaled, y_train_smtk)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = model.predict(X_test_scaled)

    # 测试集指标计算
    y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
    test_sensitivity = tp_test / (tp_test + fn_test)
    test_specificity = tn_test / (tn_test + fp_test)
    test_f1 = f1_score(y_test, y_test_pred_best)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    all_loss = sum(LOSS)

    # 构建结果字典
    result_dict = {
        'var_smoothing': params['var_smoothing'],
        'random_state_split': random_state_split,
        'random_state_cv': random_state_cv,
        'random_state_smote': random_state_smote,
        "smote_auc_scores": smote_auc_scores,
        "auc_scores_min": auc_scores_min,
        'mean_auc': mean_auc,
        'test_auc': test_auc,
        'mean_f1': mean_f1,
        'test_f1': test_f1,
        # =============== 新增指标存储 ===============
        'best_threshold': global_best_threshold,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'mean_brier': mean_brier,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_brier': test_brier,
        'smote_sensitivity_scores': smote_sensitivity_scores,
        'smote_specificity_scores': smote_specificity_scores,
        'smote_brier_scores': smote_brier_scores,
        # ==========================================
        'auc_product': mean_auc * test_auc,
        'f1_product': mean_f1 * test_f1,
        'youhua': mean_auc * test_auc * mean_f1 * test_f1,
        "loss": all_loss,
        "SORT": (mean_auc > 0.7) & (test_auc > 0.7),
        "brier": mean_brier + test_brier,
    }

    # 如果提供了auc_results列表，则将结果存入
    if auc_results is not None:
        auc_results.append(result_dict)

    # 返回hyperopt需要的格式
    return {'loss': all_loss, 'status': STATUS_OK, 'result_dict': result_dict}


