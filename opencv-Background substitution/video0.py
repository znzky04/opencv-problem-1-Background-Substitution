import cv2
import numpy as np

# 读取背景图片并调整大小以匹配视频帧的尺寸
background = cv2.imread('bookshelf.jpg')
vc = cv2.VideoCapture('video.mp4')
ret, frame = vc.read()
if frame is not None:
    background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

# 设置绿幕的HSV阈值
lower_green = np.array([40, 100, 50])
upper_green = np.array([70, 255, 255])

# 高斯模糊和羽化参数
blur_kernel_size = (15, 15)  # 高斯模糊核大小
feather_size = 21  # 羽化程度
feather_iterations = 4  # 膨胀和腐蚀的迭代次数

# 创建膨胀/腐蚀核
kernel = np.ones((feather_size, feather_size), np.uint8)

while ret:
    ret, frame = vc.read()
    if frame is None:
        break

    # 将视频帧从BGR颜色空间转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建绿幕掩码和其反向掩码
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # 对反向掩码进行高斯模糊
    mask_inv_blurred = cv2.GaussianBlur(mask_inv, blur_kernel_size, 0)

    # 使用形态学操作来羽化掩码边缘
    mask_feathered = cv2.dilate(mask_inv_blurred, kernel, iterations=feather_iterations)
    mask_feathered = cv2.erode(mask_feathered, kernel, iterations=feather_iterations)

    # 使用开运算来去除残留
    mask_feathered = cv2.morphologyEx(mask_feathered, cv2.MORPH_OPEN, kernel)

    # 归一化掩码到 0 到 1 的范围，保证数据类型为浮点数
    alpha = cv2.normalize(mask_feathered.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 将单通道 alpha 转换为三通道
    alpha = cv2.merge([alpha, alpha, alpha])

    # 使用模糊后的反向掩码提取前景人物
    foreground = cv2.bitwise_and(frame, frame, mask=mask_feathered)

    # 使用正向掩码提取背景图像中的绿幕部分
    background_portion = cv2.bitwise_and(background, background, mask=mask)

    # 手动执行加权混合
    combined = cv2.convertScaleAbs((1.0 - alpha) * background_portion + alpha * foreground)

    # 显示最终合成的结果
    cv2.imshow('result', combined)

    # 如果按下'q'键或关闭窗口则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象和销毁所有OpenCV窗口
vc.release()
cv2.destroyAllWindows()



