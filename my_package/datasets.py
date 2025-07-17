import os
import pandas as pd
from mrmr import mrmr_classif
from pathlib import Path


# 获取当前目录下的 data 文件夹路径
def prepare_data():
    # 定义文件路径
    parent_dir = Path.cwd() / "data"
    # 定义文件路径
    data_path = parent_dir / "新整合+附二.xlsx"
    features_path = parent_dir / "radiomics_features.xlsx"
    filtered_path = parent_dir / "符合筛选的特征.xlsx"

    # 示例：检查文件是否存在
    if not data_path.exists():
        print(f"文件不存在: {data_path}")

    # 加载数据
    data = pd.read_excel(data_path).rename(columns={'序号': 'case_id'})
    output01234 = pd.read_excel(features_path)
    filtered_df = pd.read_excel(filtered_path)

    # 筛选特征
    selected_vars = [var for var in filtered_df['Variable'].unique() if var in output01234.columns]
    df = output01234[selected_vars + ['case_id']]
    df['case_id'] = df['case_id'].astype(str)
    data['case_id'] = data['case_id'].astype(str)
    # 合并数据
    merged_df = pd.merge(
        df,
        data[['case_id', "gender(0=女)", "Differentiation(0=高中分化，1=低分化)",
              "Vascular invasion", "TNM stage(0=1-2期，1=3-4期)", 'ER']],
        on='case_id'
    )

    # 准备变量
    clinical_vars = ["gender(0=女)", "Differentiation(0=高中分化，1=低分化)",
                     "Vascular invasion", "TNM stage(0=1-2期，1=3-4期)"]

    # 特征选择
    y = merged_df["ER"]
    X_features = mrmr_classif(merged_df.drop(columns=['case_id', 'ER'] + clinical_vars), y, K=15)

    # 构建最终数据集
    X = merged_df[X_features + clinical_vars]
    Xr = merged_df[X_features]
    Xc = merged_df[clinical_vars]
    return X, Xr, Xc, y


def local_clinic():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path_clinic = parent_dir / "临床.xlsx"
    X = pd.read_excel(data_path_clinic).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path_clinic)["ER"]
    return X, y


def local_radiomics():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "影像.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y


def local_integrate():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "融合.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y


def local_radiomics0():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "影像0.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y


def local_radiomics1():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "影像1.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y


def local_radiomics2():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "影像2.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y


def local_radiomics3():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "影像3.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y


def local_radiomics4():
    parent_dir = Path.cwd() / "model_pre_data"
    data_path = parent_dir / "影像4.xlsx"
    X = pd.read_excel(data_path).drop(columns=['subfolder', 'ER'])
    y = pd.read_excel(data_path)["ER"]
    return X, y

