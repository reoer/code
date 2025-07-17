import os
import nrrd
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
import json
def process_nrrd_folder_structure(root_dir, output_file):
    """
    处理包含两个NRRD文件的文件夹结构，提取图像和mask的尺寸、体素空间以及mask的label值
    :param root_dir: 目标文件夹路径
    :param output_file: 输出的Excel文件路径
    """
    # 收集所有数据
    all_data = []

    # 遍历第一级子文件夹
    for folder_idx, folder_name in enumerate(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"处理文件夹 [{folder_idx + 1}]: {folder_name}")

        # 初始化变量
        mask_file = None
        image_file = None
        image_size = ""
        image_spacing = ""
        mask_size = ""
        mask_spacing = ""
        mask_labels = []
        other_files = []
        nrrd_files = []

        # 遍历子文件夹内的文件
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if os.path.isfile(file_path):
                if file.lower().endswith('.nrrd'):
                    nrrd_files.append(file)
                    # 识别mask文件(S开头)
                    if file.startswith('S') or file.startswith('s'):
                        mask_file = file
                        try:
                            # 读取NRRD文件
                            data, header = nrrd.read(file_path)

                            # 获取mask尺寸
                            mask_size = "×".join(map(str, data.shape))

                            # 获取mask体素空间
                            mask_spacing = extract_voxel_spacing(header)

                            # 提取所有label值（排除0值）
                            unique_vals = np.unique(data)
                            non_zero_vals = [v for v in unique_vals if v != 0]

                            # 提取label信息（如果有）
                            mask_labels = extract_label_info(header, non_zero_vals)

                        except Exception as e:
                            print(f"  ! 处理mask文件 {file} 时出错: {e}")
                            mask_size = f"错误: {str(e)}"
                            mask_spacing = f"错误: {str(e)}"
                            mask_labels = [{"label": f"错误: {str(e)}", "value": ""}]
                    else:
                        image_file = file
                        try:
                            # 读取图像文件
                            data, header = nrrd.read(file_path)

                            # 获取图像尺寸
                            image_size = "×".join(map(str, data.shape))

                            # 获取图像体素空间
                            image_spacing = extract_voxel_spacing(header)

                        except Exception as e:
                            print(f"  ! 处理图像文件 {file} 时出错: {e}")
                            image_size = f"错误: {str(e)}"
                            image_spacing = f"错误: {str(e)}"
                else:
                    other_files.append(file)

        # 验证NRRD文件数量
        if len(nrrd_files) != 2:
            print(f"  ! 警告: 文件夹 {folder_name} 包含 {len(nrrd_files)} 个NRRD文件 (期望2个)")

        # 构建数据行
        row_data = OrderedDict([
            ("文件夹名称", folder_name),
            ("图像文件", image_file or "未找到"),
            ("图像尺寸", image_size),
            ("图像体素空间", image_spacing),
            ("mask文件", mask_file or "未找到"),
            ("mask尺寸", mask_size),
            ("mask体素空间", mask_spacing),
            ("其他文件", ", ".join(other_files) if other_files else "无")
        ])

        # 添加mask标签信息
        for i, label_info in enumerate(mask_labels[:20]):  # 最多取20个标签
            row_data[f"Label_{i + 1}_值"] = label_info.get("value", "")
            row_data[f"Label_{i + 1}_描述"] = label_info.get("label", "")

        all_data.append(row_data)

    # 创建DataFrame
    df = pd.DataFrame(all_data)

    # 保存到Excel
    df.to_excel(output_file, index=False)
    print(f"\n处理完成！共处理 {len(all_data)} 个子文件夹")
    print(f"结果已保存至: {output_file}")



def extract_voxel_spacing(header):
    """
    从NRRD文件头中提取体素空间信息
    :param header: NRRD文件头
    :return: 格式化的体素空间字符串
    """
    # 优先尝试直接获取spacings字段
    spacings = header.get("spacings")
    if spacings is not None:
        if isinstance(spacings, list):
            return "×".join([f"{x:.4f}" for x in spacings])
        return f"{spacings:.4f}"

    # 尝试从space directions矩阵中提取
    space_directions = header.get("space directions")
    if space_directions is not None:
        # 处理不同格式的space directions
        if isinstance(space_directions, np.ndarray):
            # 提取对角线元素作为各方向间距
            spacings = [np.linalg.norm(direction) for direction in space_directions]
            return "×".join([f"{x:.4f}" for x in spacings])
        elif isinstance(space_directions, list):
            # 处理列表格式
            spacings = []
            for direction in space_directions:
                if isinstance(direction, tuple) or isinstance(direction, list):
                    # 计算向量的模作为间距
                    spacings.append(np.linalg.norm(direction))
                elif isinstance(direction, str):
                    # 尝试解析字符串格式 "(a,b,c,d)"
                    match = re.match(r"\(([\d.]+),([\d.]+),([\d.]+)\)", direction)
                    if match:
                        values = [float(x) for x in match.groups()]
                        spacings.append(np.linalg.norm(values))
            if spacings:
                return "×".join([f"{x:.4f}" for x in spacings])

    # 尝试从pixel size字段获取
    pixel_size = header.get("pixel size")
    if pixel_size is not None:
        if isinstance(pixel_size, list):
            return "×".join([f"{x:.4f}" for x in pixel_size])
        return f"{pixel_size:.4f}"

    # 最后尝试从content字段解析
    if "content" in header:
        content = header["content"]
        # 尝试匹配spacing模式
        spacing_match = re.search(r"spacing\s*[:=]\s*\(?([\d.,\s]+)\)?", content, re.IGNORECASE)
        if spacing_match:
            spacing_str = spacing_match.group(1)
            spacings = [float(x.strip()) for x in spacing_str.split(",") if x.strip()]
            if spacings:
                return "×".join([f"{x:.4f}" for x in spacings])

    return "未知"


def extract_label_info(header, values):
    """
    从NRRD文件头中提取标签信息
    :param header: NRRD文件头
    :param values: 提取的标签值列表
    :return: 包含标签描述的字典列表
    """
    labels = []

    # 尝试从标准字段获取标签信息
    if "Segmentation.Label.Value" in header:
        label_values = header["Segmentation.Label.Value"]
        label_names = header.get("Segmentation.Label.Name", [])

        # 确保是列表形式
        if not isinstance(label_values, list):
            label_values = [label_values]
        if not isinstance(label_names, list):
            label_names = [label_names]

        # 创建标签映射
        label_map = {}
        for idx, val in enumerate(label_values):
            name = label_names[idx] if idx < len(label_names) else f"标签{val}"
            label_map[val] = name

        # 为找到的值创建标签描述
        for val in values:
            labels.append({
                "value": val,
                "label": label_map.get(val, f"标签{val}")
            })
        return labels

    # 尝试从Content字段解析标签
    if "content" in header:
        content = header["content"]
        # 尝试解析JSON格式的标签
        if content.startswith('{') and content.endswith('}'):
            try:
                content_data = json.loads(content)
                if "labels" in content_data:
                    label_map = content_data["labels"]
                    for val in values:
                        labels.append({
                            "value": val,
                            "label": label_map.get(str(val), f"标签{val}")
                        })
                    return labels
            except:
                pass

    # 尝试正则表达式匹配标签描述
    if "content" in header:
        content = header["content"]
        # 匹配格式: "label: value = description"
        matches = re.findall(r'(?:label|value)\s*[=:]\s*(\d+)\s*[=:]\s*"([^"]+)"', content, re.IGNORECASE)
        if matches:
            label_map = {int(val): desc for val, desc in matches}
            for val in values:
                labels.append({
                    "value": val,
                    "label": label_map.get(val, f"标签{val}")
                })
            return labels

        # 匹配格式: "value: description"
        matches = re.findall(r'(\d+)\s*:\s*"([^"]+)"', content)
        if matches:
            label_map = {int(val): desc for val, desc in matches}
            for val in values:
                labels.append({
                    "value": val,
                    "label": label_map.get(val, f"标签{val}")
                })
            return labels

    # 默认处理：没有标签信息
    for val in values:
        labels.append({
            "value": val,
            "label": f"标签{val}"
        })
    return labels


# 使用示例
if __name__ == "__main__":
    # 输入路径设置
    target_folder = r"ICCResampled"  # 替换为你的目标文件夹路径
    output_excel = r"ICCResampled_Detailed_Analysis.xlsx"  # 替换为输出文件路径

    # 安装必要库（如果尚未安装）
    # pip install pynrrd numpy pandas openpyxl

    process_nrrd_folder_structure(target_folder, output_excel)
