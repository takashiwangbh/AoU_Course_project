import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d  # 使用 SciPy 进行高斯滤波

# 1D 高斯滤波器
def gaussian_smoothing(data, sigma=1.0):
    """
    对 1D 数据应用高斯滤波
    参数:
    - data: 输入的 1D 数据（列表或数组）
    - sigma: 高斯核的标准差（控制平滑程度）
    返回:
    - 平滑后的数据数组
    """
    return gaussian_filter1d(data, sigma=sigma)

# 读取 CSV 并应用高斯滤波
def apply_gaussian_filter(csv_file, output_file, sigma=1.0):
    try:
        # 读取 CSV
        df = pd.read_csv(csv_file)

        # 应用高斯滤波
        filtered_df = df.copy()
        for col in df.columns:
            filtered_df[col] = gaussian_smoothing(df[col].values, sigma=sigma)

        # 保存滤波后的数据
        filtered_df.to_csv(output_file, index=False)
        print(f"✅ 处理完成: {output_file}")

    except Exception as e:
        print(f"❌ 处理失败: {csv_file}，错误: {e}")

# 遍历文件夹，处理所有 CSV 文件
def process_all_csv_files(root_folder, sigma=1.0):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_filtered.csv'):  # 只处理原始 CSV
                original_file = os.path.join(root, file)
                filtered_file = os.path.join(root, file.replace('.csv', '_gaussian_filtered.csv'))  # 避免覆盖
                apply_gaussian_filter(original_file, filtered_file, sigma)

# 你的数据根目录（请替换为你的实际路径）
root_folder = r"D:\git\Course Project\AoU_Course_project\CSC02F Applied Signal Processing\Final project\sign data"

# 运行批量处理（默认 sigma=1.0，可调整平滑程度）
process_all_csv_files(root_folder, sigma=1.0)
