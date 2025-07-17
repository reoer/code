from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, atpe, rand, anneal
import numpy as np
import pandas as pd
from my_package.train import objective_MLP
from my_package.datasets import local_radiomics, local_integrate, local_clinic, \
    local_radiomics1, local_radiomics2, local_radiomics3, local_radiomics4, local_radiomics0
from functools import partial

################################################################################################
space = {
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [
        (50,), (100,), (150,), (50, 50), (100, 50), (100, 100)]),
    'activation': hp.choice('activation', ['relu', 'tanh', 'logistic']),
    'solver': hp.choice('solver', ['adam', 'sgd', 'lbfgs']),
    'alpha': hp.loguniform('alpha', np.log(1e-5), np.log(1e-1)),
    'batch_size': hp.choice('batch_size', ['auto', 16, 32, 64]),
    'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
    'learning_rate_init': hp.loguniform('learning_rate_init', np.log(1e-5), np.log(1e-1)),
    'max_iter': hp.quniform('max_iter', 100, 1000, 100),
    'early_stopping': hp.choice('early_stopping', [True, False]),
    'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-1)),
    'momentum': hp.uniform('momentum', 0.5, 0.9),
    'n_iter_no_change': hp.quniform('n_iter_no_change', 10, 50, 1),

    'random_state_cv': hp.quniform('random_state_split2', 0, 100, 1),
    'random_state_smote': hp.quniform('random_state_smote', 0, 100, 1),
    'random_state_model': hp.quniform('random_state_model', 0, 100, 1)
}
################################################################################################
trials = Trials()

#载入不同的影像数据
X, y = local_radiomics()
#X, y = local_radiomics0()
#X, y = local_radiomics1()
#X, y = local_radiomics2()
#X, y = local_radiomics3()
#X, y = local_radiomics4()
################################################################################################
auc_results = []
objective_fn = partial(objective_MLP, X=X, y=y, auc_results=auc_results)
print(X.shape)
best = fmin(fn=objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)
# 将 AUC 和 F1-score 结果保存到 Excel 文件中
auc_df = pd.DataFrame(auc_results)  # 假设 auc_results 已经定义
file_name = f'search_results/多层感知机-MRMR15.xlsx'
auc_df.to_excel(file_name, index=False)
################################################################################################
