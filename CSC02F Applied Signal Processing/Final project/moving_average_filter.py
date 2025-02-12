import numpy as np
import pandas as pd
import os

# 1D 移动平均滤波器
def moving_average_filter(data, window_size=5):
    """
    对 1D 数据应用移动平均滤波
    参数:
    - data: 输入的 1D 数据（列表或数组）
    - window_size: 滤波窗口大小（必须为奇数）
    返回:
    - 平滑后的数据数组
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 读取 CSV 并应用移动平均滤波
def apply_moving_average_filter(csv_file, output_file, window_size=5):
    try:
        # 读取 CSV
        df = pd.read_csv(csv_file)

        # 应用移动平均滤波
        filtered_df = df.copy()
        for col in df.columns:
            filtered_df[col] = moving_average_filter(df[col].values, window_size=window_size)

        # 保存滤波后的数据
        filtered_df.to_csv(output_file, index=False)
        print(f"✅ 处理完成: {output_file}")

    except Exception as e:
        print(f"❌ 处理失败: {csv_file}，错误: {e}")

# 遍历文件夹，处理所有 CSV 文件
def process_all_csv_files(root_folder, window_size=5):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_filtered.csv'):  # 只处理原始 CSV
                original_file = os.path.join(root, file)
                filtered_file = os.path.join(root, file.replace('.csv', '_moving_avg_filtered.csv'))  # 避免覆盖
                apply_moving_average_filter(original_file, filtered_file, window_size)

# 你的数据根目录（请替换为你的实际路径）
root_folder = r"D:\git\Course Project\AoU_Course_project\CSC02F Applied Signal Processing\Final project\sign data"

# 运行批量处理（窗口大小可调整，默认 5）
process_all_csv_files(root_folder, window_size=5)
