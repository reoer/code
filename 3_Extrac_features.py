import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor, getFeatureClasses


def process_directory(root_dir):
    """
    处理主目录下的所有子目录，提取PyRadiomics特征
    :param root_dir: 主目录路径
    """
    all_results = []

    # 配置PyRadiomics参数
    params = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': sitk.sitkBSpline,
        'label': 1,  # 基础标签值
        'force2D': True  # 强制处理为2D图像
    }
    # 初始化特征提取器
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('SquareRoot')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('Gradient')
    extractor.enableImageTypeByName('LocalBinaryPattern2D')
    extractor.enableImageTypeByName('LocalBinaryPattern3D')
    extractor.enableAllFeatures()
    all_features_df = pd.DataFrame()
    print("启用的滤波器类型:", extractor.enabledImagetypes.keys())
    # 获取所有特征类别的键名（用于处理None值）
    feature_classes = getFeatureClasses()
    feature_keys = []
    for cls in feature_classes.values():
        feature_keys.extend(cls.getFeatureNames())

    # 遍历子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # 查找配对的NRRD文件
        ct_path, mask_path = find_paired_nrrd_files(subdir_path)
        if not ct_path or not mask_path:
            print(f"跳过目录 {subdir} - 文件配对失败")
            continue

        print(f"处理 {subdir}:")
        print(f"  CT图像: {os.path.basename(ct_path)}")
        print(f"  Mask文件: {os.path.basename(mask_path)}")

        # 提取五类特征
        try:
            features = extract_features(extractor, ct_path, mask_path, subdir, feature_keys)
            all_results.append(features)
            print(f"  成功提取特征")
        except Exception as e:
            print(f"处理 {subdir} 时出错: {str(e)}")

    # 保存结果到Excel
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = os.path.join(root_dir, "radiomics_features.xlsx")
        df.to_excel(output_path, index=False)
        print(f"\n特征已保存至: {output_path}")
    else:
        print("未提取到有效特征")


def find_paired_nrrd_files(directory):
    """
    在目录中查找配对的NRRD文件
    :return: (ct_path, mask_path) 或 (None, None)
    """
    # 获取目录下所有NRRD文件
    all_files = os.listdir(directory)
    nrrd_files = [f for f in all_files if f.lower().endswith('.nrrd')]

    if len(nrrd_files) != 2:
        print(f"  目录中NRRD文件数量不为2（实际有{len(nrrd_files)}个），跳过")
        return None, None

    # 分离mask和CT文件
    mask_files = [f for f in nrrd_files if f.lower().startswith('s')]
    ct_files = [f for f in nrrd_files if not f.lower().startswith('s')]

    if len(mask_files) != 1 or len(ct_files) != 1:
        print(f"  无法确定配对（mask文件：{len(mask_files)}，CT文件：{len(ct_files)}）")
        return None, None

    return (
        os.path.join(directory, ct_files[0]),
        os.path.join(directory, mask_files[0])
    )


def extract_features(extractor, ct_path, mask_path, case_id, feature_keys):
    """
    提取五类特征并添加指定前缀，处理None值问题

    参数:
        extractor: PyRadiomics特征提取器
        ct_path: CT图像路径
        mask_path: 原始掩膜路径
        case_id: 案例ID
        feature_keys: 所有可能的特征键列表

    返回:
        包含所有特征的字典（每个样本一行）
    """
    # 1. 创建结果字典，包含案例ID
    results = {'case_id': case_id}

    # 2. 读取原始掩膜并创建值为5的大掩膜
    original_mask = sitk.ReadImage(mask_path)

    # 步骤1: 创建二值掩膜（1-4标签区域为1，其他为0）
    combined_mask = sitk.BinaryThreshold(original_mask,
                                         lowerThreshold=1,
                                         upperThreshold=4,
                                         insideValue=1,
                                         outsideValue=0)

    # 步骤2: 转换为值为5的掩膜
    big_mask = sitk.Cast(combined_mask, sitk.sitkUInt16) * 5

    # 3. 全区域特征提取（使用值为5的大掩膜）
    extractor.settings['label'] = 5  # 设置标签为5
    try:
        features = extractor.execute(ct_path, big_mask)
        for k, v in features.items():
            if not k.startswith('diagnostics'):
                # 处理None值
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    results[f"all_{k}"] = float('nan')
                else:
                    results[f"all_{k}"] = v
    except Exception as e:
        print(f"  警告: 全区域特征提取失败 - {str(e)}")
        for k in feature_keys:
            results[f"all_{k}"] = float('nan')

    # 4. 分割区域特征提取（使用原始掩膜）
    for i in range(1, 5):
        extractor.settings['label'] = i
        try:
            features = extractor.execute(ct_path, mask_path)
            for k, v in features.items():
                if not k.startswith('diagnostics'):
                    # 处理None值
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        results[f"split{i}_{k}"] = float('nan')
                    else:
                        results[f"split{i}_{k}"] = v
        except Exception as e:
            print(f"  警告: 标签 {i} 提取失败 - {str(e)}")
            for k in feature_keys:
                results[f"split{i}_{k}"] = float('nan')

    return results


if __name__ == "__main__":
    # 使用示例
    main_directory = "ICCResampled"  # 替换为实际路径
    process_directory(main_directory)
