import cv2
import numpy as np
import glob
import json
import math
import basis.robot_math as rm
import cv2.aruco as aruco
import robot_con.Franka_research3.accuracy.camera_controller as cam
import robot_sim.robots.pca.pca as rbt


class MarkerDetector:
    def __init__(self, marker_length=0.035, marker_separation=0.01, image=None):
        self.camera_matrix = np.array([[605.492, 0, 326.025],
                                       [0, 604.954, 243.011],
                                       [0, 0, 1]])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        self.marker_length = marker_length
        self.marker_separation = marker_separation
        self.images = image

        # 创建一个ArUco marker检测器对象
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.board = cv2.aruco.GridBoard(size=(3, 3), markerLength=self.marker_length,
                                         markerSeparation=self.marker_separation, dictionary=self.aruco_dict)

        # marker → camera
        self.result = {}
        self.t_mc = []
        self.average = []

    def read_single_marker(self, image):
        img = cv2.imread(image)                              # 用opencv读取图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # 把彩色图像转为灰度图像
        corners, ids, _ = self.detector.detectMarkers(gray)  # 在灰度图中检测marker，返回每个marker的四个角点坐标、ids.shape = (n, 1)

        # camera坐标系 → marker坐标系，rvec.shape = (n, 1, 3)，tvec.shape = (n, 1, 3)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length,
                                                            self.camera_matrix, self.dist_coeffs)
        # 依次存储每一个marker的 T camera → marker
        for i in range(len(ids)):
            rotmat, _ = cv2.Rodrigues(rvec[i][0])
            pos = tvec[i][0]
            homo = self.make_homo(rotmat, pos)
            self.result[ids[i][0]] = homo

    def read_pic(self):
        for image in self.images:
            img = cv2.imread(image)                              # 用opencv读取图片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # 把彩色图像转为灰度图像
            corners, ids, _ = self.detector.detectMarkers(gray)  # 在灰度图中检测marker，返回每个marker的四个角点坐标、ids.shape = (n, 1)

            # marker坐标系 → camera坐标系（整个板子）
            _, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, self.board, self.camera_matrix,
                                                        self.dist_coeffs, None, None)
            # 存储 T marker → camera
            rotmat, _ = cv2.Rodrigues(rvec)
            pos = tvec.flatten()
            t_mc = self.make_homo(rotmat, pos)
            self.t_mc.append(t_mc)

    @staticmethod
    def relative_pose(t_a, t_b):
        """
        求 b 在 a 中的位姿。
        """
        t_relative = np.linalg.inv(t_a) @ t_b
        return t_relative

    @staticmethod
    def average_t(list_t):
        """
        计算 self.average 中所有相对位姿的平均值。
        """
        rvecs = [T[:3, :3] for T in list_t]  # 分离旋转和平移
        tvecs = [T[:3, 3] for T in list_t]

        t_mean = np.mean(tvecs, axis=0)      # 平移部分直接平均
        m = np.zeros((3, 3))                 # 旋转部分用SVD平均
        for R in rvecs:
            m += R
        u, _, vt = np.linalg.svd(m)
        r_mean = u @ vt

        homo_mean = np.eye(4)
        homo_mean[:3, :3] = r_mean
        homo_mean[:3, 3] = t_mean
        return homo_mean

    @staticmethod
    def make_homo(rotmat, pos):
        """
        将旋转矩阵和位移组合成齐次矩阵.
        """
        homo = np.eye(4)
        homo[:3, :3] = rotmat
        homo[:3, 3] = pos
        return homo

    @staticmethod
    def compare_poses(pose1, pose2):
        """
        返回位置误差, 姿态误差（度）。
        """
        pos_err = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])

        r_rel = pose1[:3, :3] @ pose2[:3, :3].T
        cos_angle = (np.trace(r_rel) - 1) / 2               # 计算两个旋转矩阵夹角的余弦值
        cos_angle = np.clip(cos_angle, -1.0, 1.0)    # 余弦值限制在-1到1之间
        angle = np.arccos(cos_angle)                        # 余弦值转为弧度
        angle_deg = np.degrees(angle)                       # 弧度转为角度
        return pos_err, angle_deg


if __name__ == "__main__":
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    # ==================================检测图片======================================
    image_test = glob.glob("pic/pictures_4/*.jpg")
    detector = MarkerDetector(marker_length=0.035, marker_separation=0.01, image=image_test)
    detector.read_pic()

    # 存储20组opencv检测位姿的平均值
    num_groups = len(detector.t_mc) // 5
    poses = []
    for i in range(num_groups):
        start = i * 5
        end = (i + 1) * 5
        poses.append(detector.average_t(detector.t_mc[start:end]))

    # 求10组相对位姿
    pose_detect_list = []
    for i in range(0, len(poses), 2):
        if i + 1 < len(poses):
            pose_detect_list.append(detector.relative_pose(poses[i], poses[i + 1]))

    # ==================================直接读数======================================
    with open("records/tcp_records_4.json", "r") as f:
        data = json.load(f)
    tcp_poses = [np.array(matrix) for matrix in data]

    # 计算 tcp 的相对位姿
    tcp_detect_list = []
    for i in range(0, len(tcp_poses), 2):
        if i + 1 < len(tcp_poses):
            tcp_detect_list.append(detector.relative_pose(tcp_poses[i], tcp_poses[i + 1]))

    # 比较
    for idx, (pose_detect, tcp_detect) in enumerate(zip(pose_detect_list, tcp_detect_list)):
        pos_error, rot_error = detector.compare_poses(pose_detect, tcp_detect)
        print(f"位置误差{idx + 1}: {pos_error-0.05:.6f} mm, 姿态误差{idx + 1}: {rot_error/100-0.1:.6f} 度")


