import cv2
import subprocess
import numpy as np
import os
# 读取背景图片并调整大小以匹配视频帧的尺寸
background = cv2.imread('bookshelf.jpg')
vc = cv2.VideoCapture('video.mp4')
ret, frame = vc.read()
background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

# 读取音频
audio_capture = cv2.VideoCapture('audio.mp3')

# 输出视频文件名
output_video_filename = 'output_video.mp4'

# 视频编解码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 输出视频流
output_video = cv2.VideoWriter(output_video_filename, fourcc, 30, (int(frame.shape[1]), int(frame.shape[0])))

while ret:
    ret, frame = vc.read()
    if frame is None:
        break

    # 将视频帧从BGR颜色空间转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 为绿色定义HSV阈值
    lower_green = np.array([40, 100, 50])
    upper_green = np.array([70, 255, 255])

    # 创建掩码和反向掩码
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # 对掩码应用高斯模糊以软化边缘
    kernel_size = (210, 210)  # 可以调整高斯模糊的核大小
    mask_inv_blurred = cv2.GaussianBlur(mask_inv, kernel_size, 0)

    # 对背景应用相同的模糊
    background_blurred = cv2.GaussianBlur(background, kernel_size, 0)

    # 羽化掩码边缘
    feather = 23  # 通过调整此参数来设置羽化程度
    kernel = np.ones((feather, feather), np.uint8)
    mask_feathered = cv2.dilate(mask_inv_blurred, kernel, iterations=5)  # 膨胀
    mask_feathered = cv2.erode(mask_feathered, kernel, iterations=5)  # 腐蚀

    # 使用模糊后的反向掩码提取前景人物
    foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # 使用正向掩码提取背景图像中的绿幕部分
    background_portion = cv2.bitwise_and(background, background, mask=mask)

    # 合并前景和背景部分
    combined = cv2.add(foreground, background_portion)

    # 写入输出视频流
    output_video.write(combined)

# 释放VideoCapture对象和销毁所有OpenCV窗口
vc.release()
cv2.destroyAllWindows()

# 释放音频VideoCapture对象
audio_capture.release()

# 合并视频和音频文件
subprocess.run(
    ['ffmpeg', '-i', output_video_filename, '-i', 'audio.mp3', '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
     'output_final.mp4'])

# 删除临时视频文件
os.remove(output_video_filename)


