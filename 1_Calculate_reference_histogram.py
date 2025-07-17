import os
import numpy as np
import SimpleITK as sitk
import time
from tqdm import tqdm


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


def find_all_ct_files(root_dir):
    """
    查找所有CT图像文件路径
    仅包含非'S'开头的NRRD文件
    """
    ct_paths = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            file_lower = file.lower()
            # 仅处理非S开头的NRRD文件
            if (file_lower.endswith('.nrrd') and not file_lower.startswith('s')):
                file_path = os.path.join(folder_path, file)
                ct_paths.append(file_path)

    return ct_paths


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
    # 1. 检查头信息中的明确标记
    if "DICOM_RescaleType" in header:
        rescale_type = header["DICOM_RescaleType"].lower()
        if "hu" in rescale_type or "hounsfield" in rescale_type:
            return True

    # 2. 检查像素值范围特征
    min_val = np.min(img_array)
    max_val = np.max(img_array)

    # HU值的典型特征：包含负值且范围在-1000到3000之间
    if min_val < -500 and max_val > 500:
        return True

    # 3. 检查百分位数特征
    if np.percentile(img_array, 1) < -500 and np.percentile(img_array, 99) > 300:
        return True

    return False


def convert_to_hu(img_array, header):
    """将原始CT值转换为HU值"""
    slope = float(header.get("DICOM_RescaleSlope", 1.0))
    intercept = float(header.get("DICOM_RescaleIntercept", -1024.0))
    hu_array = img_array * slope + intercept
    return hu_array


def process_single_image_for_histogram(file_path):
    """处理单个图像以生成直方图"""
    try:
        image = safe_read_image(file_path)
        img_array = sitk.GetArrayFromImage(image).astype(np.float32)
        header = get_nrrd_header(file_path)

        # 转换为HU值
        if header and ('ct' in file_path.lower()):
            if not is_hu_converted(header, img_array):
                img_array = convert_to_hu(img_array, header)
            img_array = np.clip(img_array, -1000, 2000)

        # 创建直方图（忽略背景区域）
        non_background = img_array[img_array > -900]
        if len(non_background) == 0:
            return None

        hist, _ = np.histogram(non_background, bins=4096, range=(-1000, 2000))
        return hist

    except Exception as e:
        print(f"  ! 处理文件 {file_path} 时出错: {e}")
        return None
    finally:
        del image
        del img_array


def create_global_histogram(root_dir, output_path="global_reference_hist.npy"):
    """创建全局参考直方图"""
    ct_paths = find_all_ct_files(root_dir)
    if not ct_paths:
        print("警告: 未找到符合条件的CT图像文件")
        print("要求: NRRD格式，文件名包含'ct'或'ct_'，且不以'S'开头")
        return None

    print(f"找到 {len(ct_paths)} 个符合条件的CT图像文件")
    print("开始创建全局直方图...")
    start_time = time.time()

    # 顺序处理每个图像
    results = []
    for path in tqdm(ct_paths, desc="处理CT图像"):
        hist = process_single_image_for_histogram(path)
        if hist is not None:
            results.append(hist)

    if not results:
        print("错误: 未能成功处理任何图像")
        return None

    # 合并所有直方图
    global_hist = np.zeros(4096, dtype=np.float64)
    for hist in results:
        global_hist += hist

    # 归一化直方图
    global_hist_sum = np.sum(global_hist)
    if global_hist_sum > 0:
        global_hist = global_hist / global_hist_sum

    # 保存结果
    np.save(output_path, global_hist)
    print(f"全局直方图已保存至: {output_path}")
    print(f"耗时 {time.time() - start_time:.2f} 秒")
    return output_path


if __name__ == "__main__":
    input_folder = "NRRD"  # 输入文件夹路径
    output_file = "global_reference_hist.npy"  # 输出文件名

    create_global_histogram(input_folder, output_file)
