# -*- coding:utf-8 -*-

"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: ABB_Realsence_D435_capture_images.py
Description:
"""


import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from PIL import Image

save_directory = 'Data_Intel_Realsense_D435'
os.makedirs(save_directory, exist_ok=True)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imshow('Color Image', color_image)
        key = cv2.waitKey(1)
        if key == 13:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            color_image_path = os.path.join(save_directory, f'color_image_{timestamp}.jpg')
            cv2.imwrite(color_image_path, color_image)
            print(f'Saved color image as {color_image_path}')

            # 保存深度图像 (使用 Pillow)
            depth_image_path = os.path.join(save_directory, f'depth_image_{timestamp}.png')
            depth_image = depth_image.astype(np.uint16)  # 确保深度图像是 uint16 类型
            img = Image.fromarray(depth_image)
            img.save(depth_image_path)
            print(f'Saved depth image as {depth_image_path}')

            # depth_image_path = os.path.join(save_directory, f'depth_image_{timestamp}.png')
            # cv2.imwrite(depth_image_path, depth_image)
            # print(f'Saved depth image as {depth_image_path}')
        if key & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
