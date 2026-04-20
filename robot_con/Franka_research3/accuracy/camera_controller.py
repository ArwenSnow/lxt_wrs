import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
from typing import Literal, Optional


class Camera:
    def __init__(self, camera_type: Literal['d405', 'd435'] = 'd435', aruco_dict=None, aruco_params=None,
                 save_directory=None):
        self.camera_type = camera_type
        self.save_directory = save_directory

        # 确认目录存在
        os.makedirs(self.save_directory, exist_ok=True)

        # 加载内参矩阵和畸变参数
        if self.camera_type.lower() == 'd405':
            self.camera_matrix = np.array([[434.43981934, 0.0, 318.67144775],
                                           [0.0, 434.35751343, 241.73374939],
                                           [0.0, 0.0, 1.0]])
            self.dist_coeffs = np.array([[-0.05277087, 0.06000207, 0.00087849, 0.00136543, -0.01997724]])
        elif self.camera_type.lower() == 'd435':
            self.camera_matrix = np.array([[605.492, 0, 326.025],
                                           [0, 604.954, 243.011],
                                           [0, 0, 1]])
            self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError('Either d405 or d435')

        self.aruco_dict = aruco_dict or aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco_params or aruco.DetectorParameters()

        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """
        初始化并启动相机管道.
        """
        self.pipeline = rs.pipeline()                                                # 创建管道获取图像
        config = rs.config()                                                         # 创建配置对象
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)    # 启用彩色流
        self.pipeline_profile = self.pipeline.start(config)                          # 启动管道，获取配置文件
        for _ in range(5):                                                           # 丢弃前几帧使相机稳定
            self.pipeline.wait_for_frames()

    def capture_rgb(self, delay: float = 3.0, show=True) -> Optional[np.ndarray]:
        """
        拍RGB图.
        """
        captured_color_image = None
        print(f"将在 {delay} 秒后自动捕获图像...")
        start_time = time.time()

        while time.time() - start_time < delay:
            frames = self.pipeline.wait_for_frames()
            # 获取彩色帧
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
        print("正式捕获图像...")
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # 将获取的彩色帧转化成BGR格式的numpy数组
        if color_frame:
            captured_color_image = np.asanyarray(color_frame.get_data())
        # 保存图像
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
        print(f"RGB图像保存于{color_path}")
        # 保存彩色图像（BGR格式）
        cv2.imwrite(color_path, captured_color_image)

        # 显示RGB图像
        if show:
            cv2.imshow('Captured Image', captured_color_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)  # 等待用户按键
            cv2.destroyAllWindows()
        return captured_color_image

    def live_capture(self):
        """
        实时显示彩色图像，按 Enter 键保存图片，按 q 键退出。
        """
        print("相机已启动，按 Enter 键拍照，按 q 键退出...")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow('Camera Live', color_image)

                key = cv2.waitKey(1)
                if key == 13:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
                    cv2.imwrite(color_path, color_image)
                    print(f"已保存图像: {color_path}")
                elif key & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    save_dir = r"C:\Users\11154\Documents\GitHub\lxt_wrs\robot_con\Franka_research3\accuracy\pic\pictures"
    camera_r = Camera(save_directory=save_dir)
    camera_r.live_capture()

