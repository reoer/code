import pandas as pd

# 读取两个Excel文件
df1 = pd.read_excel('ICC1_Results.xlsx')  # 替换为你的第一个文件路径
df2 = pd.read_excel('ICC3_Results.xlsx')  # 替换为你的第二个文件路径

# 合并两个DataFrame，基于'Variable'列
merged_df = pd.merge(df1, df2, on='Variable', how='inner', suffixes=('_file1', '_file2'))

# 筛选ICC列都大于等于0.75的行
# 假设ICC列名分别为'ICC_file1'和'ICC_file2'，如果不是请相应调整
filtered_df = merged_df[(merged_df['ICC_file1'] >= 0.75) & (merged_df['ICC_file2'] >= 0.75)]

# 保存结果到新Excel文件（可选）
filtered_df.to_excel('符合筛选的特征.xlsx', index=False)

# 显示结果
print(filtered_df)
