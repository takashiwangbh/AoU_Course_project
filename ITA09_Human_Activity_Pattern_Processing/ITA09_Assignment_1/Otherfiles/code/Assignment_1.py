import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.ticker import MultipleLocator


def read_signature_data(file_path):
    signature_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 跳过空行
                data = line.strip().split()
                if len(data) == 6:  # 确保每行有6个值
                    signature_data.append(list(map(int, data)))
                elif len(data) == 1 and int(data[0]) == -1:  # 笔划分隔符
                    signature_data.append([-1, -999999, -999999, -999999, 999999, -999999])
    return signature_data


def draw_signature(canvas, signature_data, delay=100):
    # 设置绘制参数
    line_color = "black"
    line_width = 3

    # 记录上一个有效点的坐标和时间
    prev_x, prev_y, prev_time = None, None, None

    # 计算总时间
    total_time = signature_data[-1][-1]

    # 绘制签名
    for data in signature_data:
        if data[0] == -1:
            prev_x, prev_y, prev_time = None, None, None
            canvas.after(delay)  # 增加延迟，形成动画效果
            canvas.update()  # 更新画布
        else:
            x, y, pressure, direction, height, time = data
            if prev_x is not None and prev_y is not None and prev_time is not None:
                canvas.create_line(prev_x, prev_y, x, y, fill=line_color, width=line_width)
                canvas.after(int((time - prev_time) * delay / total_time))  # 根据时间间隔确定延迟时间
                canvas.update()  # 更新画布
            prev_x, prev_y, prev_time = x, y, time


def extract_features(signatures):
    x_coordinates = []
    y_coordinates = []
    pressures = []
    directions = []
    heights = []
    times = []

    for signature in signatures:
        for data in signature:
            if data[0] != -1:  # 确保不是笔划分隔符
                x_coordinates.append(data[0])
                y_coordinates.append(data[1])
                pressures.append(data[2])
                directions.append(data[3])
                heights.append(data[4])
                times.append(data[5])

    return x_coordinates, y_coordinates, pressures, directions, heights, times


def plot_data(ax, data, times, title):
    ax.plot(times, data)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')


# 创建主窗口
root = tk.Tk()
root.title("Signature Display Program")
root.geometry("800x800")

# 添加签名画布
canvas_signature = tk.Canvas(root, width=600, height=300, bg="white")
canvas_signature.pack(side=tk.TOP, pady=20)

# 读取签名数据文件
file_path = "D:\作业\ITA09\ITA09_Assignment_1\Otherfiles\code\SignatureSampleData\001.001.001.sdt"
signature_data = read_signature_data(file_path)

# 绘制签名
draw_signature(canvas_signature, signature_data)

# 提取特征
x_coordinates, y_coordinates, pressures, directions, heights, times = extract_features([signature_data])

# 创建图形窗口
fig, axs = plt.subplots(5, 1, figsize=(8, 20))  # 设置图形大小，使得纵向长度更长
fig.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距

# 绘制每个子图并设置纵坐标范围
plot_data(axs[0], x_coordinates, times, 'X Coordinate')
axs[0].set_ylim(min(x_coordinates) - 0.1, max(x_coordinates) + 0.1)  # 设置纵坐标范围

plot_data(axs[1], y_coordinates, times, 'Y Coordinate')
axs[1].set_ylim(min(y_coordinates) - 0.1, max(y_coordinates) + 0.1)  # 设置纵坐标范围

plot_data(axs[2], pressures, times, 'Pen Pressure')
axs[2].set_ylim(min(pressures) - 0.1, max(pressures) + 0.1)  # 设置纵坐标范围

plot_data(axs[3], directions, times, 'Pen Direction')
axs[3].set_ylim(min(directions) - 0.1, max(directions) + 0.1)  # 设置纵坐标范围

plot_data(axs[4], heights, times, 'Height')
axs[4].set_ylim(min(heights) - 0.1, max(heights) + 0.1)  # 设置纵坐标范围

plt.show()


# 运行主循环
root.mainloop()
