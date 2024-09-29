import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture('video.mp4')

# 获取视频信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 读取图像
bookshelf = cv2.imread('bookshelf.jpg')
bookshelf = cv2.resize(bookshelf, (frame_width, frame_height))  # 调整图像大小以适应视频

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    mask = cv2.inRange(frame, lower_green, upper_green)

    replaced_frame = frame.copy()
    replaced_frame[mask != 0] = bookshelf[mask != 0]

    # Color Spill Suppression
    hsv = cv2.cvtColor(replaced_frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (30, 50, 50), (90, 255, 255))  # Mask for green in HSV color space
    replaced_frame[mask_green != 0] = bookshelf[mask_green != 0]  # Replace green areas with background

    edges = cv2.Canny(mask, 100, 200)

    blurred_edges = cv2.GaussianBlur(edges, (5, 5), 0)

    mask_blurred = np.zeros_like(replaced_frame, dtype=np.float32)
    mask_blurred[blurred_edges != 0] = 1
    mask_blurred = cv2.GaussianBlur(mask_blurred, (15, 15), 0)

    alpha = 0.5
    result = cv2.addWeighted(replaced_frame, alpha, frame, 1 - alpha, 0)

    result = result * (1 - mask_blurred) + cv2.GaussianBlur(replaced_frame, (15, 15), 0) * mask_blurred

    out.write(result.astype(np.uint8))

    cv2.imshow('Result', result.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()













