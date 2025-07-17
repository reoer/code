import os

from sklearn.svm import SVC

os.environ["SCIPY_ARRAY_API"] = "1"
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, atpe, rand, anneal
from mrmr import mrmr_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, brier_score_loss, roc_curve
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
import warnings
from sklearn.utils import compute_class_weight


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # 裁剪预测值以防止 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # 计算交叉熵损失
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


warnings.filterwarnings("ignore", category=FutureWarning)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data"))
################################################################################################
data = pd.read_excel(os.path.join(parent_dir, "新整合+附二.xlsx"))
output01234 = pd.read_excel(os.path.join(parent_dir, 'radiomics_features.xlsx'))
# 从output01234中选择所有数值列 + case_id列
filtered_df = pd.read_excel(os.path.join(parent_dir, '符合筛选的特征.xlsx'))
# 获取 filtered_results.xlsx 中 Variable 列的所有唯一值
selected_variables = filtered_df['Variable'].unique()
available_vars = [var for var in selected_variables if var in output01234.columns]
# 创建要保留的列列表：选中的变量 + case_id
columns_to_keep = available_vars + ['case_id']
split_k = 5  # k折检验
test_size = 0.3  # 数据划分比例8：2
################################################################################################
auc_results = []
datanames = ["output01234"]
for dataname in datanames:
    # 筛选数据
    df = output01234[columns_to_keep]
    # 查看结果
    print(df.head())
    data = data.rename(columns={'序号': 'case_id'})
    df['case_id'] = df['case_id'].astype(str)
    data['case_id'] = data['case_id'].astype(str)
    merged_df = pd.merge(
        df,  # 数值列+case_id的DataFrame
        data[['case_id',
              "gender(0=女)",
              "Differentiation(0=高中分化，1=低分化)",
              "Vascular invasion",
              "TNM stage(0=1-2期，1=3-4期)",
              'ER']],
        left_on='case_id',
        right_on='case_id',
        how='inner'
    )

    # 提取目标变量和特征
    y = merged_df["ER"]
    X_pre = merged_df.drop(columns=["case_id", "ER"])

    # 选取指定的列
    var2 = ["gender(0=女)",
            "Differentiation(0=高中分化，1=低分化)",
            "Vascular invasion",
            "TNM stage(0=1-2期，1=3-4期)",
            ]
    # 选取指定的列

    for i in range(5, 30):
        var = mrmr_classif(X_pre.drop(columns=var2), y, K=int(i))
        selected_columns = var + var2 + ['case_id', 'ER']
        X = X_pre[var + var2]
        print("变量数据集", X.shape)
        # print("选择特征变量数量", k)
        # print("选择变量参数", selected_features)
        # selected_features = mrmr_classif(X_pre[var], y, K=30)
        X = X_pre[var + var2]
        y = merged_df["ER"]
        for k in range(1, 5):
            ##########################zz######################################################################
            random_state_split = 1027   # int(params['random_state_split'])
            random_state_split2 = 42 # #int(params['random_state_split2'])
            random_state_smote = int(k)# int(params['random_state_smote'])
            random_state_model = 42# int(params['random_state_model'])
            ################################################################################################
            # 重新划分数据集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=random_state_split,
                                                                stratify=y)
            # 数据插补
            imputer = KNNImputer(n_neighbors=5)  # 插补方法
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
            X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
            X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
            # 过采样处理
            smtk = SMOTETomek(random_state=random_state_smote)
            X_train_smtk, y_train_smtk = smtk.fit_resample(X_train, y_train)
            ################################################################################################
            """                model = MLPClassifier(
                    hidden_layer_sizes=(100,),
                    activation="tanh",
                    solver="sgd",
                    alpha=0.0003,
                    batch_size=64,
                    learning_rate="constant",
                    learning_rate_init=0.03,
                    max_iter=2000,
                    early_stopping=False,
                    tol=0.05,
                    momentum=0.9,
                    n_iter_no_change=29,
                    random_state=random_state_model,
                )"""
            model = SVC(
                C=0.07,
                kernel="poly",
                gamma="auto",
                degree=5,
                coef0=0.5,
                class_weight="balanced",  # 使用手动计算的类权重
                random_state=random_state_model,
                probability=True,  # 启用预测概率
                tol=0.03,  # 容忍度
                max_iter=40000,  # 最大迭代次数c
                shrinking=True  # 启用收缩算法
            )
            ################################################################################################
            # 标准化器
            scaler = StandardScaler()
            # =============== 新增指标收集 ===============
            smote_auc_scores = []
            smote_f1_scores = []
            smote_sensitivity_scores = []
            smote_specificity_scores = []
            smote_brier_scores = []
            best_thresholds = []  # 存储每折的最佳阈值
            all_val_probas = []  # 存储所有验证集预测概率
            all_val_y = []  # 存储所有验证集真实标签
            # ==========================================

            skf = StratifiedKFold(n_splits=split_k, random_state=random_state_split2, shuffle=True)
            ################################################################################################
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

                # =============== 新增指标计算 ===============
                # 1. 计算最佳决策阈值 (Youden's J statistic)
                fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
                youden_index = tpr - fpr
                best_idx = np.argmax(youden_index)
                best_threshold = thresholds[best_idx]
                best_thresholds.append(best_threshold)

                # 2. 使用最佳阈值计算预测
                y_val_pred_best = (y_val_proba >= best_threshold).astype(int)

                # 3. 计算敏感度(召回率)、特异度
                tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_best).ravel()
                sensitivity = tp / (tp + fn)  # 召回率/敏感度
                specificity = tn / (tn + fp)  # 特异度

                # 4. 计算F1分数
                fold_f1 = f1_score(y_val_fold, y_val_pred_best)

                # 5. 计算Brier分数
                fold_brier = brier_score_loss(y_val_fold, y_val_proba)

                # 存储当前折的指标
                smote_sensitivity_scores.append(sensitivity)
                smote_specificity_scores.append(specificity)
                smote_f1_scores.append(fold_f1)
                smote_brier_scores.append(fold_brier)

                # 收集验证集数据用于全局阈值计算
                all_val_probas.extend(y_val_proba)
                all_val_y.extend(y_val_fold)
                # ==========================================

                fold_auc = roc_auc_score(y_val_fold, y_val_proba)
                smote_auc_scores.append(fold_auc)
                loss = binary_cross_entropy(y_val_fold, y_val_proba)
                LOSS.append(loss)
            ################################################################################################
            # 计算全局最佳阈值 (使用所有验证集数据)
            fpr_all, tpr_all, thresholds_all = roc_curve(all_val_y, all_val_probas)
            youden_index_all = tpr_all - fpr_all
            global_best_threshold = thresholds_all[np.argmax(youden_index_all)]

            # 计算 AUC 和 F1 得分
            auc_scores_min = np.min(smote_auc_scores)
            mean_auc = np.mean(smote_auc_scores)
            mean_f1 = np.mean(smote_f1_scores)

            # 计算其他指标的平均值
            mean_sensitivity = np.mean(smote_sensitivity_scores)
            mean_specificity = np.mean(smote_specificity_scores)
            mean_brier = np.mean(smote_brier_scores)

            # 对测试集进行标准化
            X_test_scaled = scaler.transform(X_test)

            # 过采样后训练
            X_train_smtk_scaled = scaler.fit_transform(X_train_smtk)
            model.fit(X_train_smtk_scaled, y_train_smtk)

            # 评估测试集
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_test_pred = model.predict(X_test_scaled)

            # =============== 新增测试集指标计算 ===============
            # 使用全局最佳阈值
            y_test_pred_best = (y_test_proba >= global_best_threshold).astype(int)

            # 计算测试集敏感度、特异度
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred_best).ravel()
            test_sensitivity = tp_test / (tp_test + fn_test)
            test_specificity = tn_test / (tn_test + fp_test)

            # 计算测试集F1分数
            test_f1 = f1_score(y_test, y_test_pred_best)

            # 计算测试集Brier分数
            test_brier = brier_score_loss(y_test, y_test_proba)
            # ================================================

            test_auc = roc_auc_score(y_test, y_test_proba)
            val_loss = binary_cross_entropy(y_test, y_test_proba)
            all_loss = np.mean(LOSS) + val_loss

            auc_results.append({
                "dataname": dataname,
                'random_state_split': random_state_split,
                'random_state_split2': random_state_split2,
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
            })
            print(i, k, ":   ", mean_auc)
            # 将 AUC 和 F1-score 结果保存到 Excel 文件中
    auc_df = pd.DataFrame(auc_results)  # 假设 auc_results 已经定义
    #file_name = f'202507111003支持向量机循环影像临床.xlsx'
    #auc_df.to_excel(file_name, index=False)
    ################################################################################################
