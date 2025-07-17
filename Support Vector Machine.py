from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, atpe, rand, anneal
import numpy as np
import pandas as pd
import warnings
from my_package.train import objective_SVC
from my_package.datasets import local_radiomics, local_integrate, local_clinic, \
    local_radiomics1, local_radiomics2, local_radiomics3, local_radiomics4, local_radiomics0
from functools import partial

warnings.filterwarnings("ignore", category=FutureWarning)
################################################################################################
auc_results = []
space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(10)),  # C值范围
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),  # 核函数选择
    'gamma': hp.choice('gamma', ['scale', 'auto']),  # gamma选择
    'degree': hp.quniform('degree', 1, 10, 1),  # degree，适用于poly核函数
    'coef0': hp.uniform('coef0', 0, 10),  # coef0，适用于poly和sigmoid核函数
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-1)),  # 容忍度
    'max_iter': hp.quniform('max_iter', 30000, 50000, 1000),  # 最大迭代次数
    'shrinking': hp.choice('shrinking', [True, False]),  # 是否使用收缩算法
    'random_state_cv': hp.quniform('random_state_split2', 0, 100, 1),
    'random_state_smote': hp.quniform('random_state_smote', 0, 100, 1),
    'random_state_model': hp.quniform('random_state_model', 0, 100, 1)
}
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
################################################################################################
objective_fn = partial(objective_SVC, X=X, y=y, auc_results=auc_results)
print(X.shape)
best = fmin(fn=objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)
# 将 AUC 和 F1-score 结果保存到 Excel 文件中
auc_df = pd.DataFrame(auc_results)  # 假设 auc_results 已经定义
file_name = f'search_results/支持向量机-MRMR15.xlsx'
auc_df.to_excel(file_name, index=False)
################################################################################################
