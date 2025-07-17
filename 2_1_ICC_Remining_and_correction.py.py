import os
import numpy as np
import SimpleITK as sitk
import time
import re
import json
from tqdm import tqdm
import pypinyin
import sys
import shutil
from collections import OrderedDict
import pandas as pd
import nrrd

# 检查并导入pypinyin库
try:
    from pypinyin import lazy_pinyin, Style
except ImportError:
    print("需要安装pypinyin库，请运行: pip install pypinyin")
    sys.exit(1)


def contains_chinese(s):
    """检查字符串是否包含中文字符"""
    return any('\u4e00' <= char <= '\u9fff' for char in s)


def convert_to_pinyin(s):
    """将中文字符串转换为拼音"""
    if not contains_chinese(s):
        return s
    pinyin_list = lazy_pinyin(s, style=Style.NORMAL)
    return ''.join(pinyin_list)


def create_temp_structure(input_dir, temp_dir):
    """创建临时目录结构，将中文路径转换为拼音路径，并删除中文路径"""
    print(f"创建临时目录结构: {temp_dir}")

    # 清空或创建临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        # 计算相对路径
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == ".":
            rel_path = ""

        # 转换中文路径为拼音
        path_parts = []
        for part in rel_path.split(os.sep) if rel_path else []:
            if contains_chinese(part):
                path_parts.append(convert_to_pinyin(part))
            else:
                path_parts.append(part)

        safe_rel_path = os.path.join(*path_parts) if path_parts else ""
        temp_root = os.path.join(temp_dir, safe_rel_path)

        # 创建目标目录
        os.makedirs(temp_root, exist_ok=True)

        # 复制所有文件到临时目录
        for file in files:
            src_path = os.path.join(root, file)
            # 只处理nrrd文件
            if file.lower().endswith('.nrrd'):
                # 转换文件名（如果需要）
                if contains_chinese(file):
                    safe_file = convert_to_pinyin(os.path.splitext(file)[0]) + '.nrrd'
                else:
                    safe_file = file

                dest_path = os.path.join(temp_root, safe_file)
                shutil.copy2(src_path, dest_path)
                print(f"复制: {os.path.join(rel_path, file)} -> {os.path.join(safe_rel_path, safe_file)}")

    print(f"临时目录结构创建完成: {temp_dir}")


def safe_read_image(path, retries=3, delay=1):
    """安全读取图像，带有重试机制"""
    for attempt in range(retries):
        try:
            return sitk.ReadImage(path)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  ! 读取 {os.path.basename(path)} 失败，重试中... ({e})")
                time.sleep(delay)
            else:
                raise RuntimeError(f"无法读取图像 {path}: {e}")


def get_nrrd_header(file_path):
    """获取NRRD文件的头信息"""
    header = {}
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.ReadImageInformation()

        for key in reader.GetMetaDataKeys():
            header[key] = reader.GetMetaData(key)

        header["spacing"] = reader.GetSpacing()
        header["origin"] = reader.GetOrigin()
        header["direction"] = reader.GetDirection()

    except Exception as e:
        print(f"    ! 读取头信息失败: {e}")

    return header


def is_hu_converted(header, img_array):
    """检查是否已转换为HU值"""
    if "DICOM_RescaleType" in header:
        rescale_type = header["DICOM_RescaleType"].lower()
        if "hu" in rescale_type or "hounsfield" in rescale_type:
            return True

    min_val = np.min(img_array)
    max_val = np.max(img_array)
    if min_val < -500 and max_val > 500:
        return True

    if np.percentile(img_array, 1) < -500 and np.percentile(img_array, 99) > 300:
        return True

    return False


def convert_to_hu(img_array, header):
    """将原始CT值转换为HU值"""
    slope = float(header.get("DICOM_RescaleSlope", 1.0))
    intercept = float(header.get("DICOM_RescaleIntercept", -1024.0))
    hu_array = img_array * slope + intercept
    return hu_array


def match_histogram(source_array, reference_hist):
    """执行直方图匹配"""
    source_hist, _ = np.histogram(source_array, bins=4096, range=(-1000, 2000))
    source_cdf = source_hist.cumsum()
    source_cdf = source_cdf / source_cdf[-1]

    ref_cdf = reference_hist.cumsum()
    ref_cdf = ref_cdf / ref_cdf[-1]

    mapping = np.zeros(4096, dtype=np.float32)
    for i in range(4096):
        idx = np.argmin(np.abs(ref_cdf - source_cdf[i]))
        mapping[i] = idx

    matched_array = np.interp(
        source_array,
        np.linspace(-1000, 2000, 4096),
        mapping
    )
    return matched_array


def resample_to_isotropic(image, target_spacing=[1.0, 1.0, 1.0], is_label=False):
    """将图像重采样到各向同性体素"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)

    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)


def process_ct_file(ct_path, reference_hist, temp_dir, output_dir):
    """处理单个CT文件"""
    try:
        # 计算相对于临时目录的相对路径
        relative_path = os.path.relpath(ct_path, temp_dir)
        output_path = os.path.join(output_dir, relative_path)
        output_subdir = os.path.dirname(output_path)
        os.makedirs(output_subdir, exist_ok=True)

        ct_image = safe_read_image(ct_path)
        header = get_nrrd_header(ct_path)

        ct_array = sitk.GetArrayFromImage(ct_image)
        if header and not is_hu_converted(header, ct_array):
            ct_array = convert_to_hu(ct_array, header)
            hu_image = sitk.GetImageFromArray(ct_array)
            hu_image.CopyInformation(ct_image)
            ct_image = hu_image

        resampled_ct = resample_to_isotropic(ct_image, [1.0, 1.0, 1.0], False)
        resampled_array = sitk.GetArrayFromImage(resampled_ct)

        matched_array = match_histogram(resampled_array, reference_hist)
        matched_image = sitk.GetImageFromArray(matched_array)
        matched_image.CopyInformation(resampled_ct)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_path)
        writer.Execute(matched_image)

        print(f"已处理并保存: {relative_path}")
        return True, resampled_ct

    except Exception as e:
        print(f"  ! 处理CT文件 {os.path.basename(ct_path)} 时出错: {e}")
        return False, None


def process_mask_file(mask_path, reference_ct_image, temp_dir, output_dir):
    """处理mask文件"""
    try:
        # 计算相对于临时目录的相对路径
        relative_path = os.path.relpath(mask_path, temp_dir)
        output_path = os.path.join(output_dir, relative_path)
        output_subdir = os.path.dirname(output_path)
        os.makedirs(output_subdir, exist_ok=True)

        mask_image = safe_read_image(mask_path)
        resampled_mask = resample_to_isotropic(mask_image, [1.0, 1.0, 1.0], True)

        resampled_mask.SetOrigin(reference_ct_image.GetOrigin())
        resampled_mask.SetDirection(reference_ct_image.GetDirection())

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_path)
        writer.Execute(resampled_mask)

        print(f"已处理并保存mask: {relative_path}")
        return True

    except Exception as e:
        print(f"  ! 处理mask文件 {os.path.basename(mask_path)} 时出错: {e}")
        return False


def find_file_pairs_in_directory(dir_path):
    """在单个目录中查找CT和mask文件对"""
    files = [f for f in os.listdir(dir_path) if f.lower().endswith('.nrrd')]
    mask_files = [f for f in files if f.lower().startswith('s')]
    ct_files = [f for f in files if not f.lower().startswith('s')]

    if len(ct_files) == 1 and len(mask_files) == 1:
        ct_path = os.path.join(dir_path, ct_files[0])
        mask_path = os.path.join(dir_path, mask_files[0])
        return [(ct_path, mask_path)]
    return []


def process_all_files(input_dir, output_dir, reference_hist_path="global_reference_hist.npy", temp_dir="TEMP"):
    """
    处理所有文件：重采样、直方图匹配CT图像和mask
    步骤：
    1. 创建临时目录结构，彻底转换中文路径为拼音路径
    2. 复制原始文件到临时目录
    3. 处理临时目录中的文件
    4. 将处理结果保存到输出目录
    """
    # 创建临时目录并复制文件
    create_temp_structure(input_dir, temp_dir)

    # 加载全局参考直方图
    if not os.path.exists(reference_hist_path):
        raise FileNotFoundError(f"全局直方图文件 {reference_hist_path} 未找到")

    reference_hist = np.load(reference_hist_path)
    os.makedirs(output_dir, exist_ok=True)

    file_pairs = []
    skipped_dirs = []

    print("正在扫描临时目录结构...")
    for root, dirs, files in os.walk(temp_dir):
        if not any(f.lower().endswith('.nrrd') for f in files):
            continue

        pairs_in_dir = find_file_pairs_in_directory(root)
        if pairs_in_dir:
            file_pairs.extend(pairs_in_dir)
            print(f"在 {os.path.relpath(root, temp_dir)} 中找到文件对")
        else:
            skipped_dirs.append(root)
            print(f"  ! 跳过目录 {os.path.relpath(root, temp_dir)} - 不符合文件对规则")

    if not file_pairs:
        print("警告: 未找到有效的CT和mask文件对")
        return

    print(f"找到 {len(file_pairs)} 个有效的CT-mask文件对")
    print(f"跳过 {len(skipped_dirs)} 个不符合规则的目录")
    print("开始处理文件...")

    start_time = time.time()
    processed_ct = 0
    processed_mask = 0

    for ct_path, mask_path in tqdm(file_pairs, desc="处理文件对"):
        ct_success, resampled_ct = process_ct_file(ct_path, reference_hist, temp_dir, output_dir)
        if ct_success:
            processed_ct += 1
            if resampled_ct is not None:
                mask_success = process_mask_file(mask_path, resampled_ct, temp_dir, output_dir)
                if mask_success:
                    processed_mask += 1

    print("\n处理完成!")
    print(f"成功处理CT文件: {processed_ct}/{len(file_pairs)}")
    print(f"成功处理mask文件: {processed_mask}/{len(file_pairs)}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print(f"结果保存在: {output_dir}")
    print(f"所有图像已重采样到1×1×1 mm³体素大小")

    # 删除临时目录
    shutil.rmtree(temp_dir)
    print(f"已删除临时目录: {temp_dir}")


if __name__ == "__main__":
    input_folder = "ICC"  # 原始输入文件夹路径（可能包含中文路径）
    output_folder = "ICCResampled"  # 输出文件夹路径
    temp_folder = "TEMP"  # 临时文件夹路径（中文路径将转换为拼音）
    reference_hist_file = "global_reference_hist.npy"  # 全局直方图文件

    process_all_files(
        input_dir=input_folder,
        output_dir=output_folder,
        reference_hist_path=reference_hist_file,
        temp_dir=temp_folder
    )
