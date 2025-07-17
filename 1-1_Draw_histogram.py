import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import os
import time
import SimpleITK as sitk

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 定义器官HU值范围
# 器官HU范围定义
ORGAN_HU_RANGES = {
    '空气': (-1000, -900),
    '肺': (-900, -500),
    '脂肪': (-120, -90),
    '水': (-10, 10),
    '肝脏': (40, 60),
    '软组织': (40, 80),
    '血液': (50, 70),
    '肌肉': (35, 55),
    '肾脏': (30, 45),
    '骨(松质)': (200, 400),
    '骨(皮质)': (500, 2000),
    '主动脉': (35, 50),
    '下腔静脉': (30, 45),
    '肠道气体': (-1000, -900),
    '肠道液体': (0, 30),
    '肠壁': (30, 50),
    '淋巴结': (40, 60),
    '钙化灶': (130, 2000),
    '椎间盘': (80, 150),
    '金属植入物': (2000, 3000)
}

# 颜色映射
ORGAN_COLORS = {
    '空气': (0.8, 0.8, 1.0),
    '肺': (0.6, 0.9, 1.0),
    '脂肪': (1.0, 0.9, 0.6),
    '水': (0.6, 0.6, 1.0),
    '肝脏': (0.8, 0.6, 0.6),
    '软组织': (0.7, 0.8, 0.6),
    '血液': (1.0, 0.6, 0.6),
    '肌肉': (0.7, 0.7, 0.9),
    '肾脏': (0.6, 0.8, 0.8),
    '骨(松质)': (0.8, 0.8, 0.6),
    '骨(皮质)': (0.9, 0.8, 0.5),
    '主动脉': (1.0, 0.4, 0.4),
    '下腔静脉': (0.9, 0.5, 0.5),
    '肠道气体': (0.9, 0.9, 0.9),
    '肠道液体': (0.7, 0.7, 1.0),
    '肠壁': (0.8, 0.6, 0.7),
    '淋巴结': (0.6, 0.7, 0.6),
    '钙化灶': (0.7, 0.7, 0.7),
    '椎间盘': (0.5, 0.8, 0.5),
    '金属植入物': (0.3, 0.3, 0.3)
}

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


def load_and_analyze_histogram(npy_path, bin_range=(-1000, 2000)):
    """
    加载直方图NPY文件并进行分析
    :param npy_path: NPY文件路径
    :param bin_range: 直方图的HU值范围
    :return: (直方图数据, bin中心点, 分析结果)
    """
    # 加载直方图数据
    histogram_data = np.load(npy_path)

    # 验证数据形状
    if len(histogram_data) != 4096:
        print(f"警告: 直方图包含 {len(histogram_data)} 个bin，预期是4096")

    # 计算bin中心点
    bin_edges = np.linspace(bin_range[0], bin_range[1], len(histogram_data) + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 分析器官区域
    organ_stats = {}
    for organ, (low, high) in ORGAN_HU_RANGES.items():
        mask = (bin_centers >= low) & (bin_centers <= high)
        organ_counts = histogram_data[mask]

        if len(organ_counts) > 0:
            total = np.sum(organ_counts)
            peak_pos = bin_centers[mask][np.argmax(organ_counts)]

            organ_stats[organ] = {
                'count': total,
                'percentage': total / np.sum(histogram_data) * 100,
                'peak_hu': peak_pos,
                'bin_range': (np.min(bin_centers[mask]), np.max(bin_centers[mask]))
            }

    return histogram_data, bin_centers, organ_stats


def plot_labeled_histogram(histogram_data, bin_centers, organ_stats, figsize=(14, 7)):
    """
    绘制标注器官区域的直方图(支持中文，范围-900到1000 HU，bin使用灰黑色，标签使用黑色字体)
    :param histogram_data: 直方图数据
    :param bin_centers: 每个bin的中心HU值
    :param organ_stats: 器官统计信息
    :param figsize: 图像大小
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 确定显示范围(-900到1000 HU)
    display_mask = (bin_centers >= -900) & (bin_centers <= 1000)
    display_data = histogram_data[display_mask]
    display_bins = bin_centers[display_mask]

    # 绘制原始直方图(使用灰黑色bin)
    bar_width = display_bins[1] - display_bins[0]
    plt.bar(display_bins, display_data, width=bar_width * 0.9,
            align='center', color='#333333', edgecolor='none', label='全部像素')  # 改为灰黑色

    # 为每个器官区域添加彩色矩形(仅显示范围内的部分)
    patches = []
    for organ, stats in organ_stats.items():
        if organ in ORGAN_COLORS:
            low, high = stats['bin_range']
            # 调整显示范围不超过-900到1000
            low = max(low, -900)
            high = min(high, 1000)
            if low < high:  # 只绘制在显示范围内的部分
                rect = Rectangle((low, 0), high - low, np.max(display_data) * 1.05,
                                 alpha=0.3, color=ORGAN_COLORS[organ])
                patches.append(rect)

                # 注释掉矩形掩膜中的文字标签
                # mid_point = (low + high) / 2
                # if -900 <= mid_point <= 1000:
                #     plt.text(mid_point, np.max(display_data)*0.9, organ,
                #             ha='center', va='top', rotation=90, color='black',
                #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                #             fontsize=10)

    # 添加彩色区域
    pc = PatchCollection(patches, match_original=True)
    ax.add_collection(pc)

    # 标记峰值(仅显示范围内的峰值)
    for organ, stats in organ_stats.items():
        if organ in ORGAN_COLORS and -900 <= stats['peak_hu'] <= 1000:
            idx = np.argmin(np.abs(display_bins - stats['peak_hu']))
            plt.scatter(stats['peak_hu'], display_data[idx],
                        color=ORGAN_COLORS[organ], s=50, zorder=5, label=f'{organ}峰值')

    # 设置标题和坐标轴标签(黑色字体)
    plt.title('CT直方图 - 器官区域标注(-900到1000 HU)', fontsize=14, pad=20, color='black')
    plt.xlabel('亨氏单位 (HU)', fontsize=12, color='black')
    plt.ylabel('像素计数', fontsize=12, color='black')

    # 设置坐标轴刻度颜色
    ax.tick_params(axis='both', colors='black')

    plt.grid(True, alpha=0.3)

    # 设置x轴范围
    plt.xlim(-900, 1000)

    # 添加图例(黑色字体) - 保留右侧的label框
    legend_organs = [org for org in organ_stats.keys()
                     if organ_stats[org]['bin_range'][0] <= 1000 and
                     organ_stats[org]['bin_range'][1] >= -900]
    legend_elements = [Rectangle((0, 0), 1, 1, color=ORGAN_COLORS[org], alpha=0.3, label=org)
                       for org in legend_organs if org in ORGAN_COLORS]

    if legend_elements:
        legend = plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                            title='器官区域', title_fontsize=12, fontsize=10)
        # 设置图例文本颜色为黑色
        plt.setp(legend.get_texts(), color='black')
        plt.setp(legend.get_title(), color='black')

    plt.tight_layout()
    plt.show()


def print_organ_statistics(organ_stats):
    """打印器官统计信息"""
    print("\n器官像素统计:")
    print("-" * 60)
    print(f"{'器官':<15}{'像素数量':>15}{'占比(%)':>10}{'峰值HU':>10}{'HU范围':>15}")
    print("-" * 60)

    for organ, stats in organ_stats.items():
        print(f"{organ:<15}{stats['count']:>15,}{stats['percentage']:>10.2f}"
              f"{stats['peak_hu']:>10.1f}{str(stats['bin_range']):>15}")
    print("-" * 60)
    total_pixels = sum([s['count'] for s in organ_stats.values()])
    print(f"{'总计':<15}{total_pixels:>15,}{100:>10.2f}{'':>25}")


def main(npy_path):
    """主函数"""
    # 1. 加载并分析直方图
    histogram_data, bin_centers, organ_stats = load_and_analyze_histogram(npy_path)

    # 2. 打印统计信息
    print_organ_statistics(organ_stats)

    # 3. 绘制标注直方图
    plot_labeled_histogram(histogram_data, bin_centers, organ_stats)


if __name__ == "__main__":
    # 使用示例
    npy_file = "global_reference_hist.npy"  # 替换为你的NPY文件路径
    if os.path.exists(npy_file):
        main(npy_file)
    else:
        print(f"错误: 文件 {npy_file} 不存在")
        print("请提供有效的NPY文件路径")
