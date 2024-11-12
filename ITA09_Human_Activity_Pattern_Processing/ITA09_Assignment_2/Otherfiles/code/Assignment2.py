import numpy as np
import matplotlib.pyplot as plt


def matching_distance(A, B):
    m = len(A)
    n = len(B)

    # 初始化距离矩阵
    dp = np.zeros((m + 1, n + 1))

    # 初始化第一行和第一列
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + abs(A[i - 1] - B[0])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + abs(A[0] - B[j - 1])

    # 计算其余位置的距离
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = abs(A[i - 1] - B[j - 1]) + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp


def backtrack_matching(dp, A, B):
    m, n = dp.shape
    i = m - 1
    j = n - 1
    matching_points = []
    while i > 0 and j > 0:
        matching_points.append((i - 1, j - 1))  # 将A和B之间的对应点加入列表中
        if dp[i - 1][j - 1] <= dp[i - 1][j] and dp[i - 1][j - 1] <= dp[i][j - 1]:
            i -= 1
            j -= 1
        elif dp[i - 1][j] <= dp[i - 1][j - 1] and dp[i - 1][j] <= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return matching_points[::-1]  # 反转列表，使其按照A的顺序排列


# 读取数据集 A
data_a_path = r"database/data_a.txt"
with open(data_a_path, 'r') as file:
    data_a = [float(line.strip()) for line in file.readlines()]

# 读取数据集 B
data_b_path = r"database/data_b.txt"
with open(data_b_path, 'r') as file:
    data_b = [float(line.strip()) for line in file.readlines()]

# 计算匹配距离
dp = matching_distance(data_a, data_b)

# 回溯找到对应点
matching_points = backtrack_matching(dp, data_a, data_b)

# 绘制图像
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 绘制原始图像
axs[0].plot(data_a, color='red', label='A')
axs[0].plot(data_b, color='blue', label='B')

# 标记对应点
# for i, j in matching_points:
#     axs[0].plot([i, j], [data_a[i], data_b[j]], color='green')
axs[0].legend()
axs[0].set_title('Original')

# 更改 B 的水平坐标
new_data_b = [None] * len(data_a)
for i, j in matching_points:
    new_data_b[i] = data_b[j]

# 将未匹配到的点用前一个匹配点填充
for i in range(len(new_data_b)):
    if new_data_b[i] is None:
        new_data_b[i] = new_data_b[i - 1]

# 绘制更改后的图像
axs[1].plot(data_a, color='red', label='A')
axs[1].plot(new_data_b, color='blue', label='B')

# 标记对应点
axs[1].legend()
axs[1].set_title('Modified')

plt.show()
