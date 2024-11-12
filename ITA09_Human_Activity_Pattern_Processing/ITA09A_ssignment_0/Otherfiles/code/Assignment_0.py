import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 导入 Pandas 库
from sklearn.neighbors import KNeighborsClassifier

# 训练数据文件夹路径
train_data_folder = "D:\作业\ITA09\ITA09A_ssignment_0\Otherfiles\code\date"

# 动作列表
actions = ["act01", "act02", "act03"]

# Task 1: Read all train data. And plot figure of each data. Each figure will contain 3 axis data

for action in actions:
    action_folder = os.path.join(train_data_folder, action)
    for i in range(1, 8):
        file_path = os.path.join(action_folder, f"{i:02d}.txt")
        # 读取数据文件
        with open(file_path, 'r') as file:
            data = file.readlines()
            # 解析数据
            data = [list(map(float, line.strip().split('\t'))) for line in data]  # 使用制表符分隔
            data = list(zip(*data))  # 转置数据以便于绘图

            # 绘制图形
            plt.figure(figsize=(8, 6))
            plt.plot(data[0], color='red', label='X')
            plt.plot(data[1], color='green', label='Y')
            plt.plot(data[2], color='blue', label='Z')
            plt.title(f'{action} - {i:02d}')
            plt.legend()
            plt.show()

# Task 2: Extract Root Mean Square (RMS) from each axis of each data and generate the feature vector for training. The feature vector size will be 7x3 for walking, 7x3 for sitting, and 7x3 for jogging.

# 初始化特征向量列表
feature_vectors = []

# 读取并提取每个数据的特征
for action in actions:
    action_features = []  # 存储当前动作的特征向量
    for i in range(1, 8):
        file_path = os.path.join(train_data_folder, action, f"{i:02d}.txt")
        # 读取数据文件
        with open(file_path, 'r') as file:
            data = file.readlines()
            # 解析数据
            data = [list(map(float, line.strip().split('\t'))) for line in data]  # 使用制表符分隔
            data = np.array(data)

            # 计算每个轴的均方根值
            rms_values = np.sqrt(np.mean(data ** 2, axis=0))

            # 将均方根值添加到当前动作的特征向量列表中
            action_features.append(rms_values)

    # 将当前动作的特征向量列表添加到总特征向量列表中
    feature_vectors.append(action_features)

# 将特征向量转换为 NumPy 数组
feature_vectors = np.array(feature_vectors)

# 将所有的训练数据特征向量合并为一个大的特征向量
X_train = np.concatenate(feature_vectors)

# 创建训练标签
y_train = np.array(["walking"] * 7 + ["sitting"] * 7 + ["jogging"] * 7)

# Task 3: Plot the training feature in 3D plot as follows:

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制训练数据的特征向量
for i in range(len(X_train)):
    x, y, z = X_train[i]
    label = y_train[i]
    ax.scatter(x, y, z, label=label)

# 设置图形标题和标签
ax.set_xlabel("RMS Feature of x-axis")
ax.set_ylabel("RMS Feature of y-axis")
ax.set_zlabel("RMS Feature of z-axis")

# 显示图例
ax.legend()

# 显示图形
plt.show()

# Task 4: Extract RMS feature from each sample test data.

# 测试数据文件夹路径
test_data_folder = r"D:\作业\ITA09\ITA09A_ssignment_0\Otherfiles\code\Test"

# 初始化测试特征向量列表
test_feature_vectors = []

# 读取并提取每个测试数据的特征
for i in range(1, 11):
    file_path = os.path.join(test_data_folder, f"{i:02d}.txt")
    # 读取数据文件
    with open(file_path, 'r') as file:
        data = file.readlines()
        # 解析数据
        data = [list(map(float, line.strip().split('\t'))) for line in data]  # 使用制表符分隔
        data = np.array(data)

        # 计算每个轴的均方根值
        rms_values = np.sqrt(np.mean(data ** 2, axis=0))

        # 将均方根值添加到测试特征向量列表中
        test_feature_vectors.append(rms_values)

# 将测试特征向量转换为 NumPy 数组
test_feature_vectors = np.array(test_feature_vectors)

# 打印测试特征向量的形状
print("Shape of test_feature_vectors:", test_feature_vectors.shape)

# Task 5: Finally, classify all test sample using k-nearest neighbor classification algorithm.

# 创建k最近邻分类器
knn = KNeighborsClassifier(n_neighbors=3)

# 拟合模型
knn.fit(X_train, y_train)

# 进行预测
y_pred = knn.predict(test_feature_vectors)

# 打印预测结果
for i, pred in enumerate(y_pred):
    print(f"Sample {i+1}: {pred}")

# 打印预测结果
results_df = pd.DataFrame({"Sample": np.arange(1, 11), "Prediction": y_pred})

# 创建表格图像
plt.figure(figsize=(8, 6))
plt.table(cellText=results_df.values,
          colLabels=results_df.columns,
          cellLoc='center',
          loc='center')
plt.axis('off')  # 关闭坐标轴
plt.title('Prediction Results')
plt.show()
