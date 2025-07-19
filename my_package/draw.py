import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

def plot_roc_curve(metrics_list, save_path, title):
    plt.figure(figsize=(10, 10))
    # Define custom colors for the first three plots
    custom_colors = [
        (243 / 255, 188 / 255, 74 / 255),  # RGB 243,188,74
        (236 / 255, 102 / 255, 43 / 255),  # RGB 236,102,43
        (141 / 255, 46 / 255, 22 / 255)  # RGB 141,46,22
    ]

    for i, metrics in enumerate(metrics_list):
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        for fpr, tpr in zip(metrics.fpr_list, metrics.tpr_list):
            # 预处理：确保每个fpr/tpr以(0,0)开头
            if fpr[0] != 0 or tpr[0] != 0:
                fpr = np.concatenate([[0.0], fpr])
                tpr = np.concatenate([[0.0], tpr])
            # 移除重复的fpr并保留最大tpr
            fpr_tpr = {}
            for fp, tp in zip(fpr, tpr):
                if fp in fpr_tpr:
                    if tp > fpr_tpr[fp]:
                        fpr_tpr[fp] = tp
                else:
                    fpr_tpr[fp] = tp
            sorted_fpr = np.array(sorted(fpr_tpr.keys()))
            sorted_tpr = np.array([fpr_tpr[fp] for fp in sorted_fpr])
            # 插值处理
            interp_tpr = np.interp(mean_fpr, sorted_fpr, sorted_tpr)
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = np.mean(metrics.smote_auc_scores)

        # 强制添加(0,0)到绘图数据
        plot_fpr = np.concatenate([[0], mean_fpr])
        plot_tpr = np.concatenate([[0], mean_tpr])

        # Use custom color if available, otherwise use default
        if i < len(custom_colors):
            plt.plot(plot_fpr, plot_tpr,label=f'{metrics.model_name} Mean ROC (AUC = {mean_auc:.2f})',linewidth=2,color=custom_colors[i])
        else:
            plt.plot(plot_fpr, plot_tpr,label=f'{metrics.model_name} Mean ROC (AUC = {mean_auc:.2f})',linewidth=2)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(title, fontsize=24)
    plt.legend(loc='lower right', fontsize=19)
    plt.savefig(save_path)
    plt.close()


# ----------- DeLong检验工具函数 -----------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1  # +1调整为1-based索引
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


from scipy.stats import norm  # Import norm directly from scipy.stats


def calc_pvalue(aucs, sigma):
    # 计算 z 值
    l = np.array([[1, -1]])  # 定义对比矩阵
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))  # 计算 z 值
    return 2 * (1 - norm.cdf(z))  # 使用 scipy.stats.norm.cdf


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = np.vstack(
        (predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)[0][0]


from scipy.interpolate import interp1d


def plot_test_roc_curves(metrics_list, save_path, title):
    # ====== 数据校验 ======
    ref_actual = metrics_list[0].test_predictions['Actual']
    for m in metrics_list[1:]:
        if not np.array_equal(m.test_predictions['Actual'], ref_actual):
            raise ValueError("测试集标签不一致，无法进行DeLong检验")

    # ====== 计算两两p值 ======
    n_models = len(metrics_list)
    p_matrix = np.full((n_models, n_models), np.nan)
    model_names = [m.model_name for m in metrics_list]

    for i in range(n_models):
        for j in range(i + 1, n_models):
            y_true = metrics_list[i].test_predictions['Actual'].values
            y_prob_i = metrics_list[i].test_predictions['Predictedvalue'].values
            y_prob_j = metrics_list[j].test_predictions['Predictedvalue'].values
            p_matrix[i, j] = delong_roc_test(y_true, y_prob_i, y_prob_j)

    # ====== 创建颜色方案 ======
    custom_colors = [
        np.array([118, 192, 171]) / 255.0,
        np.array([58, 133, 186]) / 255.0,
        np.array([33, 45, 133]) / 255.0
    ]
    remaining_colors = plt.cm.tab10(np.linspace(0, 1, max(0, len(metrics_list) - 3)))
    colors = custom_colors + remaining_colors.tolist()[:max(0, len(metrics_list) - 3)]

    # ====== 1. 保存纯ROC曲线图 ======
    plt.figure(figsize=(10, 10))
    for idx, metrics in enumerate(metrics_list):
        y_true = metrics.test_predictions['Actual']
        y_prob = metrics.test_predictions['Predictedvalue']
        fpr, tpr, _ = roc_curve(y_true, y_prob, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        # 预处理：确保每个fpr/tpr以(0,0)开头
        if fpr[0] != 0 or tpr[0] != 0:
            fpr = np.concatenate([[0.0], fpr])
            tpr = np.concatenate([[0.0], tpr])
        # 移除重复的fpr并保留最大tpr
        fpr_tpr = {}
        for fp, tp in zip(fpr, tpr):
            if fp in fpr_tpr:
                if tp > fpr_tpr[fp]:
                    fpr_tpr[fp] = tp
            else:
                fpr_tpr[fp] = tp
        sorted_fpr = np.array(sorted(fpr_tpr.keys()))
        sorted_tpr = np.array([fpr_tpr[fp] for fp in sorted_fpr])
        # 插值到100个点
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, sorted_fpr, sorted_tpr)
        # 强制添加(0,0)以确保起始点
        plot_fpr = np.concatenate([[0], mean_fpr])
        plot_tpr = np.concatenate([[0], interp_tpr])

        plt.plot(plot_fpr, plot_tpr, color=colors[idx], lw=2, label=f'{model_names[idx]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(title, fontsize=24)
    plt.legend(loc="lower right", fontsize=19)
    plt.savefig(save_path)
    plt.close()

    # ====== 2. 保存带DeLong检验表格的图 ======
    plt.figure(figsize=(14, 10))
    for idx, metrics in enumerate(metrics_list):
        y_true = metrics.test_predictions['Actual']
        y_prob = metrics.test_predictions['Predictedvalue']
        fpr, tpr, _ = roc_curve(y_true, y_prob, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        # 预处理：确保每个fpr/tpr以(0,0)开头
        if fpr[0] != 0 or tpr[0] != 0:
            fpr = np.concatenate([[0.0], fpr])
            tpr = np.concatenate([[0.0], tpr])
        # 移除重复的fpr并保留最大tpr
        fpr_tpr = {}
        for fp, tp in zip(fpr, tpr):
            if fp in fpr_tpr:
                if tp > fpr_tpr[fp]:
                    fpr_tpr[fp] = tp
            else:
                fpr_tpr[fp] = tp
        sorted_fpr = np.array(sorted(fpr_tpr.keys()))
        sorted_tpr = np.array([fpr_tpr[fp] for fp in sorted_fpr])
        # 插值到100个点
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, sorted_fpr, sorted_tpr)
        # 强制添加(0,0)以确保起始点
        plot_fpr = np.concatenate([[0], mean_fpr])
        plot_tpr = np.concatenate([[0], interp_tpr])

        plt.plot(plot_fpr, plot_tpr, color=colors[idx], lw=2, label=f'{model_names[idx]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('ROC Curves with DeLong Test', fontsize=24)
    plt.legend(loc="lower right", fontsize=19)

    # 添加DeLong检验表格
    cell_text = []
    for i in range(n_models):
        row = []
        for j in range(n_models):
            if j > i:
                p = p_matrix[i, j]
                row.append(f'{p:.3f}' if p >= 0.001 else '<0.001')
            else:
                row.append('—')
        cell_text.append(row)

    table = plt.table(cellText=cell_text,
                      rowLabels=model_names,
                      colLabels=model_names,
                      loc='right',
                      bbox=[1.1, 0.15, 0.5, 0.7],
                      cellLoc='center')
    table.auto_set_font_size(False)
    for key, cell in table.get_celld().items():
        cell.set_fontsize(5)
    plt.subplots_adjust(right=0.8)

    delong_path = f"{os.path.splitext(save_path)[0]}_delong{os.path.splitext(save_path)[1]}"
    plt.savefig(delong_path, bbox_inches='tight', dpi=300)
    plt.close()
def plot_dca_curve(metrics_list, save_path, title):
    plt.figure(figsize=(10, 10))
    thresholds = np.linspace(0, 1, 100)
    for metrics in metrics_list:
        plt.plot(thresholds, metrics.net_benefit_test, label=f'{metrics.model_name} Test DCA Curve', linewidth=2)

    plt.plot([0, 1], [0, 0], color='gray', linestyle='--', label='No Model', linewidth=2)
    plt.plot(thresholds, metrics_list[0].net_benefit_all_positive, color='g', linestyle='--',
             label='All Positive Model', linewidth=2)
    plt.xlabel('Threshold Probability', fontsize=20)
    plt.ylabel('Net Benefit', fontsize=20)
    plt.title(title, fontsize=24)
    plt.legend(loc='lower right', fontsize=19)
    plt.ylim(-1, 1)
    plt.savefig(save_path)
    plt.close()