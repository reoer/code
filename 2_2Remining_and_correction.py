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
    """更可靠地检查是否已转换为HU值"""
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


def match_histogram(source_array, reference_hist):
    """执行直方图匹配"""
    # 计算源图像的累积分布函数（CDF）
    source_hist, _ = np.histogram(source_array, bins=4096, range=(-1000, 2000))
    source_cdf = source_hist.cumsum()
    source_cdf = source_cdf / source_cdf[-1]  # 归一化

    # 计算参考直方图的CDF
    ref_cdf = reference_hist.cumsum()
    ref_cdf = ref_cdf / ref_cdf[-1]  # 归一化

    # 创建映射函数
    mapping = np.zeros(4096, dtype=np.float32)
    for i in range(4096):
        # 找到最接近的参考CDF值
        idx = np.argmin(np.abs(ref_cdf - source_cdf[i]))
        mapping[i] = idx

    # 应用映射
    matched_array = np.interp(
        source_array,
        np.linspace(-1000, 2000, 4096),
        mapping
    )

    return matched_array


def resample_to_isotropic(image, target_spacing=[1.0, 1.0, 1.0], is_label=False):
    """将图像重采样到各向同性体素（1x1x1 mm³）

    参数:
        image: 输入的SimpleITK图像
        target_spacing: 目标体素间距 (mm)
        is_label: 是否为标签图像（决定插值方法）

    返回:
        重采样后的SimpleITK图像
    """
    # 获取原始图像的空间参数
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    # 计算新的图像尺寸（保持物理尺寸不变）
    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]

    # 设置重采样参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)

    # 设置插值方法
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 标签图像使用最近邻插值
        resampler.SetDefaultPixelValue(0)  # 背景值设为0
    else:
        resampler.SetInterpolator(sitk.sitkLinear)  # CT图像使用线性插值
        resampler.SetDefaultPixelValue(-1000)  # 背景值设为空气HU

    resampler.SetTransform(sitk.Transform())

    # 执行重采样
    resampled_image = resampler.Execute(image)

    return resampled_image


def process_ct_file(ct_path, reference_hist, input_dir, output_dir):
    """处理单个CT文件：重采样、直方图匹配并保存"""
    try:
        # 创建输出子目录（保持原始结构）
        relative_path = os.path.relpath(ct_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        output_subdir = os.path.dirname(output_path)
        os.makedirs(output_subdir, exist_ok=True)

        # 读取CT图像
        ct_image = safe_read_image(ct_path)
        header = get_nrrd_header(ct_path)

        # 转换为HU值（如果需要）
        ct_array = sitk.GetArrayFromImage(ct_image)
        if header and not is_hu_converted(header, ct_array):
            ct_array = convert_to_hu(ct_array, header)
            hu_image = sitk.GetImageFromArray(ct_array)
            hu_image.CopyInformation(ct_image)
            ct_image = hu_image

        # 重采样到1x1x1 mm³（线性插值）
        resampled_ct = resample_to_isotropic(ct_image, [1.0, 1.0, 1.0], is_label=False)

        # 获取重采样后的数组
        resampled_array = sitk.GetArrayFromImage(resampled_ct)

        # 应用直方图匹配
        matched_array = match_histogram(resampled_array, reference_hist)

        # 创建匹配后的图像
        matched_image = sitk.GetImageFromArray(matched_array)
        matched_image.CopyInformation(resampled_ct)  # 复制重采样后的空间信息

        # 保存匹配后的图像
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_path)
        writer.Execute(matched_image)

        print(f"已处理并保存: {relative_path}")
        return True, resampled_ct  # 返回重采样后的CT图像作为mask的参考

    except Exception as e:
        print(f"  ! 处理CT文件 {os.path.basename(ct_path)} 时出错: {e}")
        return False, None


def process_mask_file(mask_path, reference_ct_image, input_dir, output_dir):
    """处理mask文件：重采样到参考CT图像的空间参数（最近邻插值）"""
    try:
        # 创建输出子目录（保持原始结构）
        relative_path = os.path.relpath(mask_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        output_subdir = os.path.dirname(output_path)
        os.makedirs(output_subdir, exist_ok=True)

        # 读取mask图像
        mask_image = safe_read_image(mask_path)

        # 重采样mask到参考CT图像的空间参数（最近邻插值）
        resampled_mask = resample_to_isotropic(mask_image, [1.0, 1.0, 1.0], is_label=True)

        # 确保mask与CT具有相同的空间参数
        resampled_mask.SetOrigin(reference_ct_image.GetOrigin())
        resampled_mask.SetDirection(reference_ct_image.GetDirection())

        # 保存重采样后的mask
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
    # 获取目录中的所有NRRD文件
    files = [f for f in os.listdir(dir_path) if f.lower().endswith('.nrrd')]

    # 过滤出mask文件（以's'开头，不区分大小写）
    mask_files = [f for f in files if f.lower().startswith('s')]

    # 过滤出CT文件（不以's'开头，不区分大小写）
    ct_files = [f for f in files if not f.lower().startswith('s')]

    # 检查文件数量是否符合预期
    if len(ct_files) == 1 and len(mask_files) == 1:
        ct_path = os.path.join(dir_path, ct_files[0])
        mask_path = os.path.join(dir_path, mask_files[0])
        return [(ct_path, mask_path)]

    # 如果不符合预期，返回空列表
    return []


def process_all_files(input_dir, output_dir, reference_hist_path="global_reference_hist.npy"):
    """处理所有文件：重采样、直方图匹配CT图像和mask"""
    # 加载全局参考直方图
    if not os.path.exists(reference_hist_path):
        raise FileNotFoundError(f"全局直方图文件 {reference_hist_path} 未找到")

    reference_hist = np.load(reference_hist_path)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有文件对（CT和对应的mask）
    file_pairs = []
    skipped_dirs = []

    print("正在扫描目录结构...")
    for root, dirs, files in os.walk(input_dir):
        # 跳过没有NRRD文件的目录
        if not any(f.lower().endswith('.nrrd') for f in files):
            continue

        # 在当前目录中查找文件对
        pairs_in_dir = find_file_pairs_in_directory(root)

        if pairs_in_dir:
            file_pairs.extend(pairs_in_dir)
            print(f"在 {os.path.relpath(root, input_dir)} 中找到文件对")
        else:
            skipped_dirs.append(root)
            print(f"  ! 跳过目录 {os.path.relpath(root, input_dir)} - 不符合文件对规则")

    if not file_pairs:
        print("警告: 未找到有效的CT和mask文件对")
        return

    print(f"找到 {len(file_pairs)} 个有效的CT-mask文件对")
    print(f"跳过 {len(skipped_dirs)} 个不符合规则的目录")
    print("开始处理文件...")

    start_time = time.time()
    processed_ct = 0
    processed_mask = 0
    failed_ct = 0
    failed_mask = 0

    # 处理每个文件对
    for ct_path, mask_path in tqdm(file_pairs, desc="处理文件对"):
        # 处理CT文件
        ct_success, resampled_ct = process_ct_file(ct_path, reference_hist, input_dir, output_dir)
        if ct_success:
            processed_ct += 1

            # 处理mask文件（使用重采样后的CT作为参考）
            if resampled_ct is not None:
                mask_success = process_mask_file(mask_path, resampled_ct, input_dir, output_dir)
                if mask_success:
                    processed_mask += 1
                else:
                    failed_mask += 1
                    print(f"  ! mask处理失败: {os.path.basename(mask_path)}")
            else:
                failed_mask += 1
                print(f"  ! 缺少参考图像，无法处理mask: {os.path.basename(mask_path)}")
        else:
            failed_ct += 1
            print(f"  ! CT处理失败: {os.path.basename(ct_path)}")
            # 即使CT失败也尝试处理mask（使用原始CT作为参考）
            try:
                ct_image = safe_read_image(ct_path)
                mask_success = process_mask_file(mask_path, ct_image, input_dir, output_dir)
                if mask_success:
                    processed_mask += 1
                else:
                    failed_mask += 1
                    print(f"  ! mask处理失败: {os.path.basename(mask_path)}")
            except:
                failed_mask += 1
                print(f"  ! 无法读取CT作为参考，mask处理失败: {os.path.basename(mask_path)}")

    print("\n处理完成!")
    print(f"成功处理CT文件: {processed_ct}/{len(file_pairs)}")
    print(f"成功处理mask文件: {processed_mask}/{len(file_pairs)}")
    print(f"失败的CT文件: {failed_ct}, 失败的mask文件: {failed_mask}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print(f"结果保存在: {output_dir}")
    print(f"所有图像已重采样到1×1×1 mm³体素大小")


if __name__ == "__main__":
    input_folder = "NRRD"  # 输入文件夹路径
    output_folder = "Resampled"  # 输出文件夹路径
    reference_hist_file = "global_reference_hist.npy"  # 全局直方图文件

    process_all_files(input_folder, output_folder, reference_hist_file)