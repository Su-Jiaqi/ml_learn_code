import cv2
import os

# 打开视频文件
video_path = "v2.mp4"
cap = cv2.VideoCapture(video_path)

# 创建保存帧变化图像的文件夹
output_folder = "frame_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 每24帧抽取一帧并处理
frame_skip = 24
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % frame_skip == 0:
        resized_frame = cv2.resize(frame, (32, 32))
        output_path = os.path.join(output_folder, f"frame_{frame_index:04d}.png")
        cv2.imwrite(output_path, resized_frame)

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
