import cv2
import subprocess
import numpy as np
import os

cap = cv2.VideoCapture('video.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

bookshelf = cv2.imread('bookshelf.jpg')
bookshelf = cv2.resize(bookshelf, (frame_width, frame_height))

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

ret, frame = cap.read()

audio_capture = cv2.VideoCapture('audio.mp3')

output_video_filename = 'output_video.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_video = cv2.VideoWriter(output_video_filename, fourcc, 30, (int(frame.shape[1]), int(frame.shape[0])))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    mask = cv2.inRange(frame, lower_green, upper_green)

    replaced_frame = frame.copy()
    replaced_frame[mask != 0] = bookshelf[mask != 0]

    edges = cv2.Canny(mask, 100, 200)

    blurred_edges = cv2.GaussianBlur(edges, (3, 3), 0)

    result = replaced_frame.copy()
    result[blurred_edges != 0] = cv2.GaussianBlur(replaced_frame, (11, 11), 0)[blurred_edges != 0]  # 增加高斯模糊的半径

    out.write(result)



# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

# 释放音频VideoCapture对象
audio_capture.release()

# 合并视频和音频文件
subprocess.run(['ffmpeg', '-i', output_video_filename, '-i', 'audio.mp3', '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', 'output_final.mp4'])

# 删除临时视频文件
os.remove(output_video_filename)

# 完成
print("Video with audio merged successfully!")





