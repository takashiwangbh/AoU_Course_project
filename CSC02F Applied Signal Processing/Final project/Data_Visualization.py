import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r"D:\git\Course Project\AoU_Course_project\CSC02F Applied Signal Processing\Final project\sign data\Wang_action\Best\middle_gaussian_filtered.csv")
data = df.values

# 计算所有关键点的速度（相邻帧之间的欧氏距离）
velocities = np.sqrt(np.sum(np.diff(data, axis=0)**2, axis=1))

# 绘制速度曲线
plt.figure(figsize=(15, 5))
plt.plot(velocities)
plt.title('Vitrul_woman_action-Best-Middle')
plt.xlabel('Frame rate')
plt.ylabel('speed')
plt.grid(True)
plt.show()

# 打印一些基本信息
print(f"总帧数: {len(data)}")
print(f"数据形状: {data.shape}")