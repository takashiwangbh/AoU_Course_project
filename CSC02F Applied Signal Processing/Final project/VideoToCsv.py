import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

def process_video(video_path, hands):
    """处理单个视频文件并提取手部关键点"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"警告: 无法打开视频文件: {video_path}")
        return None
    
    # 用于存储关键点的列表
    keypoints_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用tqdm创建进度条
    for _ in tqdm(range(total_frames), desc=f"处理视频: {os.path.basename(video_path)}"):
        success, frame = cap.read()
        if not success:
            break
        
        # 将BGR图像转换为RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像并检测手部
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame_keypoints = []
                for landmark in hand_landmarks.landmark:
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
                keypoints_list.append(frame_keypoints)
    
    cap.release()
    return keypoints_list

def save_to_csv(keypoints_list, output_path):
    """将关键点数据保存为CSV文件"""
    if keypoints_list and len(keypoints_list) > 0:
        df = pd.DataFrame(keypoints_list)
        df.to_csv(output_path, index=False)
        print(f"CSV文件已保存: {output_path}")
        return True
    return False

def process_directory(input_dir):
    """递归处理目录中的所有视频文件"""
    # 支持的视频文件扩展名
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # 初始化MediaPipe手部模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 统计处理结果
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    # 遍历目录
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                output_path = os.path.splitext(video_path)[0] + '.csv'
                
                # 如果CSV文件已存在，跳过处理
                if os.path.exists(output_path):
                    print(f"跳过已处理的文件: {file}")
                    continue
                
                try:
                    print(f"\n开始处理视频: {file}")
                    keypoints_list = process_video(video_path, hands)
                    
                    if keypoints_list and save_to_csv(keypoints_list, output_path):
                        processed_count += 1
                    else:
                        failed_count += 1
                        failed_files.append(video_path)
                        print(f"警告: 视频处理失败或未检测到手部关键点: {file}")
                
                except Exception as e:
                    failed_count += 1
                    failed_files.append(video_path)
                    print(f"错误: 处理视频时出现异常: {file}")
                    print(f"错误信息: {str(e)}")
    
    # 释放资源
    hands.close()
    
    # 打印处理总结
    print("\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    
    if failed_files:
        print("\n处理失败的文件:")
        for file in failed_files:
            print(f"- {file}")

if __name__ == "__main__":
    # 设置输入目录
    input_directory = r'C:\Users\Takashiwangbh\Desktop\Singlunguagevideo_vitrul'
    
    # 确认目录存在
    if not os.path.exists(input_directory):
        print(f"错误: 目录不存在: {input_directory}")
        exit(1)
    
    # 开始处理
    print(f"开始处理目录: {input_directory}")
    process_directory(input_directory)