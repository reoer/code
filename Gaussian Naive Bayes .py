from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, atpe, rand, anneal
import numpy as np
import pandas as pd
import warnings
from my_package.train import  objective_NB
from my_package.datasets import local_radiomics, local_integrate, local_clinic, \
    local_radiomics1, local_radiomics2, local_radiomics3, local_radiomics4, local_radiomics0
from functools import partial

################################################################################################
space = {
        'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-9), np.log(1e-1)),  # 平滑参数
        'random_state_cv': hp.quniform('random_state_cv', 0, 100, 1),
        'random_state_smote': hp.quniform('random_state_smote', 0, 100, 1),
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
objective_fn = partial(objective_NB, X=X, y=y, auc_results=auc_results)
print(X.shape)
best = fmin(fn=objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)
# 将 AUC 和 F1-score 结果保存到 Excel 文件中
auc_df = pd.DataFrame(auc_results)  # 假设 auc_results 已经定义
file_name = f'search_results/朴素贝叶斯-MRMR15.xlsx'
auc_df.to_excel(file_name, index=False)
################################################################################################
