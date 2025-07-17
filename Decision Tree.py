from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, atpe, rand, anneal
import numpy as np
import pandas as pd
import warnings
from my_package.train import objective_DT
from my_package.datasets import local_radiomics, local_integrate, local_clinic, \
    local_radiomics1, local_radiomics2, local_radiomics3, local_radiomics4, local_radiomics0
from functools import partial
space = {
        'criterion': hp.choice('criterion', ['gini', 'entropy']),  # 分割标准
        'splitter': hp.choice('splitter', ['best', 'random']),  # 分割策略
        'max_depth': hp.quniform('max_depth', 1, 60, 1),  # 最大深度
        'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),  # 内部节点分割所需的最小样本数
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 60, 1),  # 叶子节点所需的最小样本数
        'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),  # 叶子节点所需的最小权重比例
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),  # 最大特征数
        'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 1),  # 最大叶子节点数
        'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.5),  # 分割导致的不纯度减少的最小值
        'class_weight': hp.choice('class_weight', [None, 'balanced']),  # 类别权重
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 0.1),  # 复杂度参数，用于剪枝
        'random_state_cv': hp.quniform('random_state_cv', 0, 100, 1),
        'random_state_smote': hp.quniform('random_state_smote', 0, 100, 1),
        'random_state_model': hp.quniform('random_state_model', 100, 200, 1)
}
################################################################################################
trials = Trials()
#载入不同的数据
X, y = local_clinic()
#X, y = local_radiomics()
#X, y = local_radiomics0()
#X, y = local_radiomics1()
#X, y = local_radiomics2()
#X, y = local_radiomics3()
#X, y = local_radiomics4()
#X, y = local_integrate()
auc_results = []
objective_fn = partial(objective_DT, X=X, y=y, auc_results=auc_results)
print(X.shape)
best = fmin(fn=objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)
# 将 AUC 和 F1-score 结果保存到 Excel 文件中
auc_df = pd.DataFrame(auc_results)  # 假设 auc_results 已经定义
file_name = f'search_results/决策树-MRMR15.xlsx'
auc_df.to_excel(file_name, index=False)
################################################################################################
