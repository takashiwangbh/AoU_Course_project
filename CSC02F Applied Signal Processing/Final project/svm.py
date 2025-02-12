import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d

# ==========================
# 🔹 处理不同信号方法 🔹
# ==========================

# ✅ 1. 卡尔曼滤波
def kalman_filter_1d(data, process_noise=1e-4, measurement_noise=1e-2, initial_estimate=0, initial_error=1):
    estimate = initial_estimate  
    error_estimate = initial_error  
    filtered_data = []
    
    for measurement in data:
        error_estimate += process_noise
        kalman_gain = error_estimate / (error_estimate + measurement_noise)
        estimate = estimate + kalman_gain * (measurement - estimate)
        error_estimate = (1 - kalman_gain) * error_estimate
        filtered_data.append(estimate)

    return np.array(filtered_data)

# ✅ 2. 移动平均滤波
def moving_average_filter(data, window_size=7):  
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# ✅ 3. 高斯滤波
def gaussian_smoothing(data, sigma=1.5):  
    return gaussian_filter1d(data, sigma=sigma)

# ==========================
# 🔹 读取 & 处理 CSV 文件 🔹
# ==========================

def get_csv_files(root_folder, filter_type):
    """
    获取特定类型的 CSV 文件
    """
    csv_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                # 根据 filter_type 选择合适的文件
                if filter_type == "original" and "_" not in file:  # 原始数据没有 "_filtered"
                    csv_files.append(os.path.join(root, file))
                elif filter_type == "kalman" and file.endswith("_filtered.csv"):
                    csv_files.append(os.path.join(root, file))
                elif filter_type == "moving_avg" and file.endswith("_moving_avg_filtered.csv"):
                    csv_files.append(os.path.join(root, file))
                elif filter_type == "gaussian" and file.endswith("_gaussian_filtered.csv"):
                    csv_files.append(os.path.join(root, file))

    if len(csv_files) == 0:
        raise ValueError(f"❌ 没有找到匹配 {filter_type} 的 CSV 文件，请检查文件命名！")

    print(f"✅ 找到 {len(csv_files)} 个 {filter_type} 数据集: {csv_files}")
    return csv_files

def load_and_merge_csv(file_list, filter_type):
    """
    读取并合并 CSV 数据，应用对应的信号处理
    """
    all_data = []
    labels = []
    
    for file in file_list:
        try:
            df = pd.read_csv(file)
            print(f"📂 正在处理: {file}, 形状: {df.shape}")

            # 确保数据是 100 帧为一个样本
            num_samples = len(df) // 100
            if num_samples == 0:
                print(f"⚠️ 文件 {file} 数据不足 100 行，跳过")
                continue

            df = df.iloc[:num_samples * 100]
            X_samples = df.values.reshape(num_samples, -1)

            # 处理数据（不同滤波方法）
            if filter_type == "kalman":
                X_samples = np.apply_along_axis(kalman_filter_1d, 1, X_samples)
            elif filter_type == "moving_avg":
                X_samples = np.apply_along_axis(moving_average_filter, 1, X_samples)
            elif filter_type == "gaussian":
                X_samples = np.apply_along_axis(gaussian_smoothing, 1, X_samples)

            # 获取类别标签
            label = os.path.basename(file).split("_")[0]
            y_labels = [label] * num_samples

            all_data.append(X_samples)
            labels.extend(y_labels)

        except Exception as e:
            print(f"❌ 读取 {file} 失败: {e}")

    if len(all_data) == 0:
        raise ValueError("❌ 没有找到有效数据！")

    X = np.vstack(all_data)
    y = np.array(labels)

    return X, y

# ==========================
# 🔹 训练 SVM 分类模型 🔹
# ==========================

def train_svm(X, y):
    # 数据归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集 & 测试集（80% 训练，20% 测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 训练 SVM
    svm_model = SVC(kernel='rbf', C=10, gamma=0.1)  
    svm_model.fit(X_train, y_train)

    # 预测 & 计算准确率
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ {filter_type.upper()} 处理后的 SVM 预测准确度: {accuracy:.2%}")

# ==========================
# 🔹 运行流程 🔹
# ==========================

root_folder = r"D:\git\Course Project\AoU_Course_project\CSC02F Applied Signal Processing\Final project\sign data"

# 选择数据处理方式（修改这里即可）
# filter_type = "original"   # 原始数据
# filter_type = "kalman"    # 卡尔曼滤波
# filter_type = "moving_avg" # 移动平均滤波
filter_type = "gaussian"   # 高斯滤波

# 获取 CSV 文件
csv_files = get_csv_files(root_folder, filter_type)

# 读取 & 处理数据
X, y = load_and_merge_csv(csv_files, filter_type)

# 训练 SVM 并输出分类准确度
train_svm(X, y)
