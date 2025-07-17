import pandas as pd
import pingouin as pg

# 读取Excel文件
#file_path = "iccradiomics_features.xlsx"  # 替换为你的Excel文件路径
file_path = "data\\纵向拼接结果.xlsx"
df = pd.read_excel(file_path, header=0)  # 第一行作为列名

# ====== 修改部分开始 ======
# 根据case_id分离观察者数据
observer1_df = df[~df['case_id'].str.endswith(('2', '3'))].copy()
observer2_df = df[df['case_id'].str.endswith('3')].copy()

# 为观察者2创建匹配的case_id（去掉末尾的"2"）
observer2_df.loc[:, 'base_id'] = observer2_df['case_id'].str[:-1]

# 合并两个观察者的数据（基于base_id匹配）
merged = observer1_df.merge(observer2_df,
                            left_on='case_id',
                            right_on='base_id',
                            suffixes=('_obs1', '_obs2'))
print(len(merged))

# 验证匹配结果
assert len(merged) == len(observer1_df), "观察者数据匹配不完整"
print(f"共检测到 {len(merged)} 个评估对象")

# ====== 构建长格式数据 ======
long_data = []
variables = [col for col in df.columns if col != 'case_id']  # 排除case_id列

for var in variables:
    for idx, row in merged.iterrows():
        # 获取观察者1和观察者2的数据
        obs1_val = row[f'{var}_obs1']
        obs2_val = row[f'{var}_obs2']

        long_data.append({
            "Object": idx + 1,
            "Rater": "Observer1",
            "Score": obs1_val,
            "Variable": var
        })
        long_data.append({
            "Object": idx + 1,
            "Rater": "Observer2",
            "Score": obs2_val,
            "Variable": var
        })

long_df = pd.DataFrame(long_data)
# ====== 修改部分结束 ======

# 确保Score是数值类型
long_df['Score'] = pd.to_numeric(long_df['Score'], errors='coerce')

# 检查转换结果
print("\n数据类型检查:")
print(long_df.dtypes)

# 计算ICC
results = []
for var in variables:  # 只计算有效变量
    subset = long_df[long_df["Variable"] == var]

    if subset['Score'].isnull().all():
        print(f"警告: 变量 '{var}' 所有值均为NaN，跳过计算")
        continue

    icc = pg.intraclass_corr(
        data=subset,
        targets="Object",
        raters="Rater",
        ratings="Score",
        nan_policy="omit"
    ).round(4)

    # 提取ICC结果
    try:
        icc_result = icc[icc["Type"] == "ICC3"].iloc[0]
        results.append({
            "Variable": var,
            "ICC": icc_result["ICC"],
            "CI95%_lower": icc_result["CI95%"][0],
            "CI95%_upper": icc_result["CI95%"][1]
        })
    except IndexError:
        print(f"错误: 无法计算变量 '{var}' 的ICC，可能数据不足")

# 输出结果
if results:
    results_df = pd.DataFrame(results)
    print("\nICC(3,1) 计算结果:")
    print(results_df)
    results_df.to_excel("ICC3_Results.xlsx", index=False)
    print("结果已保存")
else:
    print("错误: 没有可计算的ICC结果")
