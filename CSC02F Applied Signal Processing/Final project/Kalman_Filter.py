import numpy as np
import pandas as pd
import os

# 调整后的 1D 卡尔曼滤波器
def kalman_filter_1d(data, process_noise=1e-4, measurement_noise=1e-2, initial_estimate=0, initial_error=1):
    """
    调整后的卡尔曼滤波，保持信号形状但减少噪声
    """
    estimate = initial_estimate  # 初始估计值
    error_estimate = initial_error  # 初始误差
    filtered_data = []

    for measurement in data:
        # 预测步骤
        error_estimate += process_noise  # 误差增加（调整 Q 值）

        # 更新步骤
        kalman_gain = error_estimate / (error_estimate + measurement_noise)  # 计算卡尔曼增益
        estimate = estimate + kalman_gain * (measurement - estimate)  # 更新估计值
        error_estimate = (1 - kalman_gain) * error_estimate  # 更新误差

        filtered_data.append(estimate)

    return np.array(filtered_data)

# 读取 CSV 并应用卡尔曼滤波
def apply_kalman_filter(csv_file, output_file):
    try:
        # 读取 CSV
        df = pd.read_csv(csv_file)

        # 应用调整后的卡尔曼滤波
        filtered_df = df.copy()
        for col in df.columns:
            filtered_df[col] = kalman_filter_1d(df[col].values)

        # 保存滤波后的数据
        filtered_df.to_csv(output_file, index=False)
        print(f"✅ 处理完成: {output_file}")

    except Exception as e:
        print(f"❌ 处理失败: {csv_file}，错误: {e}")

# 遍历文件夹，处理所有 CSV 文件
def process_all_csv_files(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_filtered.csv'):  # 只处理原始 CSV
                original_file = os.path.join(root, file)
                filtered_file = os.path.join(root, file.replace('.csv', '_filtered.csv'))
                apply_kalman_filter(original_file, filtered_file)

# 你的数据根目录（请替换为你的实际路径）
root_folder = r"D:\git\Course Project\AoU_Course_project\CSC02F Applied Signal Processing\Final project\sign data"

# 批量处理所有 CSV 文件
process_all_csv_files(root_folder)
