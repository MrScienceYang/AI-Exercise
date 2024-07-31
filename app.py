from flask import Flask, request, make_response, jsonify  # 导入Flask框架和相关模块，用于处理HTTP请求和响应
from flask_cors import CORS# 导入Flask-CORS模块，允许跨域请求
import cv2  # 导入OpenCV库，用于图像处理
import mediapipe as mp  # 导入MediaPipe库，用于人体姿态检测

import io  # 导入io模块，用于处理字节流
import os  # 导入os模块，用于文件操作
import tempfile  # 导入tempfile模块，用于创建临时文件
from moviepy.editor import ImageSequenceClip  # 导入moviepy库中的ImageSequenceClip，用于将图像序列合成为视频

app = Flask(__name__)  # 创建一个Flask应用实例
CORS(app)  # 启用跨域请求支持
mp_drawing = mp.solutions.drawing_utils  # MediaPipe绘图工具，用于绘制人体姿态标记
mp_pose = mp.solutions.pose  # MediaPipe Pose模块，用于检测人体姿态

counter = 0  # 初始化计数器，记录俯卧撑数量
status = True  # 初始化状态，True表示在上方位置，False表示在下方位置

def process_frame(frame):
    global counter, status  # 使用全局变量

    # 将帧转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用MediaPipe Pose处理帧
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)  # 处理帧，获取姿态检测结果
        if results.pose_landmarks:  # 如果检测到人体姿态标记
            # 在帧上绘制姿态标记和连接线
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            # 根据姿态标记的位置计算俯卧撑数量
            landmarks = results.pose_landmarks.landmark
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
            right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y

            # 根据肩膀和肘部的位置确定是否处于上方或下方位置
            if left_elbow_y < left_shoulder_y and right_elbow_y < right_shoulder_y:
                if not status:  # 从下方位置移动到上方位置，计数器加一
                    counter += 1
                    status = True
            else:
                status = False  # 处于下方位置

            # 在帧上绘制俯卧撑计数
            cv2.putText(frame, f'Push-ups: {counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame # 返回处理后的帧和计数器


@app.route('/process_video', methods=['POST'])
def process_video():
    OUTPUT_DIR = 'output'  # 输出文件夹
    if 'file' not in request.files:  # 如果请求中没有文件
        return {'error': 'No file part'}, 400  # 返回错误响应
    file = request.files['file']  # 获取上传的文件
    video_stream = io.BytesIO(file.read())  # 将文件读入字节流

    video_filename = os.path.join(OUTPUT_DIR, 'uploaded_video.mp4')  # 上传视频的临时文件名
    with open(video_filename, 'wb') as temp_file:
        temp_file.write(video_stream.getvalue())  # 将字节流写入临时文件
    cap = cv2.VideoCapture(video_filename)  # 打开视频文件
    processed_frames = []  # 存储处理后的帧
    counters = 0  # 初始化计数器
    while True:
        ret, frame = cap.read()  # 读取视频帧
        if not ret:  # 如果没有读取到帧，退出循环
            break
        processed_frame = process_frame(frame)  # 处理帧
        processed_frames.append(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))  # 将处理后的帧添加到列表中
    counters = counter  # 更新计数器
    cap.release()  # 释放视频捕获对象
    os.remove(video_filename)  # 删除临时文件

    # 将处理后的帧合成为视频
    processed_video_filename = os.path.join(OUTPUT_DIR, 'processed_video.mp4')
    clip = ImageSequenceClip([frame for frame in processed_frames], fps=25)
    clip.write_videofile(processed_video_filename, codec='libx264', audio=False)  # 写入处理后的视频文件

    video_url = "http://127.0.0.1:5000/video/processed_video.mp4"  # 处理后的视频URL

    return jsonify({'video_url': video_url, 'counters': counters})  # 返回视频URL和计数器


@app.route('/video/<string:filename>', methods=['GET'])
def get_img(filename):
    if filename is None:
        pass
    else:
        image_data = open("output/" + filename, "rb").read()  # 读取视频文件数据
        response = make_response(image_data)  # 创建响应
        response.headers['Content-Type'] = 'video/mp4'  # 设置响应头
        return response  # 返回响应


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # 运行Flask应用，启用调试模式，端口为5000
